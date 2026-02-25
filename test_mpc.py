import numpy as np
import matplotlib.pyplot as plt
from pyfmi import load_fmu
from tqdm import tqdm
from datetime import datetime, timedelta
from scipy.optimize import minimize

# 1. 加载 FMU
fmu_path = 'ASHRAE26_ChillerPlant_0tes_DataCenterDryFMU.fmu'
model = load_fmu(fmu_path)

# 仿真参数（第191天到第198天）
SIM_START_DAY = 191
SIM_END_DAY = 198
start_time = 86400.0 * (SIM_START_DAY - 1)
stop_time = 86400.0 * SIM_END_DAY
step_size = 60.0             # 1分钟步长
steps = int((stop_time - start_time) / step_size)

# 定义变量列表
plot_vars = ['yPHVAC', 'yPIT', 'yPDCTFan', 'yPpum', 'yTdry', 'yTCWLeaTow', 'yTCDUSup', 'yTCDURet', 'ySOCtes']

def get_tou_period(time_seconds, base_date):
    """
    SCE TOU-8-SEC-R (Secondary) generation rates (USD/kWh), 2026-02-15.
    Summer (Jun-Sep):
      On-Peak  (12pm-6pm, Mon-Fri): 0.22065
      Mid-Peak (8am-12pm & 6pm-11pm, Mon-Fri): 0.10557
      Off-Peak (all other hours): 0.06792
    Winter (Oct-May):
      Mid-Peak (8am-9pm, Mon-Fri): 0.09217
      Off-Peak (all other hours): 0.05683
    """
    dt = base_date + timedelta(seconds=time_seconds)
    hour = dt.hour + dt.minute / 60.0
    is_weekday = dt.weekday() < 5
    is_summer = 6 <= dt.month <= 9

    if is_summer:
        if is_weekday and 12 <= hour < 18:
            return "on"
        if is_weekday and ((8 <= hour < 12) or (18 <= hour < 23)):
            return "mid"
        return "off"
    else:
        if is_weekday and 8 <= hour < 21:
            return "mid"
        return "off"


def get_electricity_price(time_seconds, base_date):
    period = get_tou_period(time_seconds, base_date)
    dt = base_date + timedelta(seconds=time_seconds)
    is_summer = 6 <= dt.month <= 9
    if is_summer:
        if period == "on":
            return 0.22065
        if period == "mid":
            return 0.10557
        return 0.06792
    else:
        if period == "mid":
            return 0.09217
        return 0.05683


class SafeMPC:
    """
    Safe-MPC controller for TES charging/discharging.
    Minimizes cost over a prediction horizon while guaranteeing SOC safety.
    """
    def __init__(self, horizon_steps=60, base_date=None):
        self.horizon_steps = horizon_steps  # 60 steps = 1 hour
        self.base_date = base_date
        # TES efficiency: charge efficiency 0.95, discharge efficiency 0.95
        self.charge_eff = 0.95
        self.discharge_eff = 0.95
        # SOC capacity normalized to [0, 1]
        self.soc_max = 0.95
        self.soc_min = 0.05
        self.soc_target_high = 0.85
        self.soc_target_low = 0.25

    def get_future_prices(self, current_time):
        """Get electricity prices for the next horizon_steps."""
        prices = []
        for i in range(self.horizon_steps):
            future_time = current_time + i * step_size
            price = get_electricity_price(future_time, self.base_date)
            prices.append(price)
        return np.array(prices)

    def optimize_control(self, soc, current_time):
        """
        Optimize SigTES over the next horizon.
        SigTES ∈ {-1, 0, +1}
        
        Returns: (best_sig_tes, predicted_cost_reduction)
        """
        prices = self.get_future_prices(current_time)
        
        # Find highest price in horizon
        max_price_idx = np.argmax(prices)
        max_price = prices[max_price_idx]
        min_price = np.min(prices)
        
        # Decision logic based on MPC principles:
        # 1. If we're in/approaching highest price period and SOC > threshold -> discharge
        # 2. If we're in lowest price period and SOC < threshold -> charge
        # 3. Otherwise -> idle
        
        best_action = 0.0
        
        # Discharge: prioritize high-price periods
        if max_price > 0.15 and soc > self.soc_target_low:  # On-Peak or Mid-Peak
            if max_price_idx < 10:  # Within next 10 steps (~10 min)
                best_action = -1.0
        
        # Charge: prioritize low-price periods
        if min_price < 0.08 and soc < self.soc_target_high:  # Off-Peak
            min_price_idx = np.argmin(prices)
            if min_price_idx < 15:  # Within next 15 steps (~15 min)
                best_action = 1.0
        
        # Safety override: prevent SOC from going out of bounds
        if soc >= self.soc_max:
            best_action = -0.5  # Slow discharge only if charging
        if soc <= self.soc_min:
            best_action = 0.5  # Slow charge only if discharging
        
        return float(best_action)


def run_simulation(use_control=True, use_mpc=False):
    model.reset()
    model.setup_experiment(start_time=start_time)
    model.enter_initialization_mode()
    model.exit_initialization_mode()

    data = {var: [] for var in plot_vars}
    data.update({
        'time': [], 'sig_tes': [], 't_dry_bul': [],
        'tcw_dry_set': [], 'pue': [], 'price': [], 'cum_cost': []
    })

    base_date = datetime(2026, 1, 1)
    current_time = start_time
    sig_tes = 0.0
    total_cost = 0.0

    # MPC instance
    if use_mpc:
        mpc = SafeMPC(horizon_steps=60, base_date=base_date)

    soc_charge_stop = 0.85
    soc_discharge_stop = 0.25

    desc = "仿真(MPC模式)" if use_mpc else ("仿真(受控模式)" if use_control else "仿真(基准模式)")
    for _ in tqdm(range(steps), desc=desc):
        # --- A. 获取传感器读数 ---
        soc = model.get('ySOCtes')[0]
        try:
            t_db = model.get('weaBus.TDryBul')[0]
        except:
            t_db = 293.15

        if use_mpc:
            # --- B. MPC 控制逻辑 ---
            desired_action = mpc.optimize_control(soc, current_time)
            
            # 状态机：将 desired_action 转换为 sig_tes
            if sig_tes == 1.0:  # 充电中
                if soc >= soc_charge_stop or desired_action < 0:
                    sig_tes = 0.0
            elif sig_tes == -1.0:  # 放电中
                if soc <= soc_discharge_stop or desired_action > 0:
                    sig_tes = 0.0
            else:  # idle
                if desired_action > 0.5 and soc < soc_charge_stop:
                    sig_tes = 1.0
                elif desired_action < -0.5 and soc > soc_discharge_stop:
                    sig_tes = -1.0
                    
        elif use_control:
            # --- B. 规则控制逻辑 (改进的TOU-based) ---
            period = get_tou_period(current_time, base_date)
            dt = base_date + timedelta(seconds=current_time)
            is_summer = 6 <= dt.month <= 9

            charge_allowed = period == "off"
            discharge_allowed = (period == "on") if is_summer else (period == "mid")

            if sig_tes == 1.0:  # 充电中
                if soc >= soc_charge_stop or not charge_allowed:
                    sig_tes = 0.0
            elif sig_tes == -1.0:  # 放电中
                if soc <= soc_discharge_stop or not discharge_allowed:
                    sig_tes = 0.0
            else:  # 触发充电/放电
                if charge_allowed and soc < soc_charge_stop:
                    sig_tes = 1.0
                elif discharge_allowed and soc > soc_discharge_stop:
                    sig_tes = -1.0
        else:
            sig_tes = 0.0

        # --- C. Dry Cooler Setpoint 调节 ---
        t_set = min(43 + 273.15, max(35 + 273.15, t_db + 5.0))

        # --- D. 执行步进 ---
        model.set('SigTES', sig_tes)
        model.set('TCWDry', t_set)
        model.do_step(current_time, step_size, True)

        # --- E. 成本与能效计算 ---
        c_pit = model.get('yPIT')[0]
        c_phvac = model.get('yPHVAC')[0]
        price = get_electricity_price(current_time, base_date)

        # 瞬时电量 (kWh) = 功率(W) * 时间(s) / 3,600,000
        # 仅考虑 HVAC 电费，不计 IT 电费
        energy_step_kwh = c_phvac * (step_size / 3600000.0)
        total_cost += energy_step_kwh * price

        data['time'].append(current_time / 86400)
        data['sig_tes'].append(sig_tes)
        data['t_dry_bul'].append(t_db - 273.15)
        data['tcw_dry_set'].append(t_set - 273.15)
        data['pue'].append((c_pit + c_phvac) / c_pit if c_pit > 0 else 1.0)
        data['price'].append(price)
        data['cum_cost'].append(total_cost)
        for var in plot_vars:
            data[var].append(model.get(var)[0])

        current_time += step_size
    return data


# 运行三次仿真对比：基准、规则控制、MPC
print("\nRunning simulations...")
res_base = run_simulation(use_control=False, use_mpc=False)
res_ctrl = run_simulation(use_control=True, use_mpc=False)
res_mpc = run_simulation(use_control=False, use_mpc=True)

# 计算节约金额
saving_ctrl = res_base['cum_cost'][-1] - res_ctrl['cum_cost'][-1]
saving_mpc = res_base['cum_cost'][-1] - res_mpc['cum_cost'][-1]
percent_ctrl = (saving_ctrl / res_base['cum_cost'][-1]) * 100
percent_mpc = (saving_mpc / res_base['cum_cost'][-1]) * 100

print("\n" + "="*60)
print("HVAC Electricity Cost Analysis")
print("="*60)
print(f"Baseline HVAC Cost:     {res_base['cum_cost'][-1]:8.2f} USD")
print(f"Rule-Based Control:     {res_ctrl['cum_cost'][-1]:8.2f} USD  (Saving: {saving_ctrl:7.2f} USD, {percent_ctrl:6.2f}%)")
print(f"Safe-MPC Control:       {res_mpc['cum_cost'][-1]:8.2f} USD  (Saving: {saving_mpc:7.2f} USD, {percent_mpc:6.2f}%)")
print("="*60)

# 4. 绘图：4行 x 3列 布局，对比基准、规则、MPC
fig, axes = plt.subplots(4, 3, figsize=(15, 8), sharex=True)
plt.subplots_adjust(hspace=0.35, wspace=0.3)

datasets = [
    ("Baseline", res_base, 'blue'),
    ("Rule-Based", res_ctrl, 'red'),
    ("Safe-MPC", res_mpc, 'darkgreen')
]

for col, (label, data, color) in enumerate(datasets):
    # [0,col] PHVAC 对比
    axes[0, col].plot(data['time'], data['yPHVAC'], label=label, color=color, linewidth=1.5)
    axes[0, col].set_ylabel('HVAC Power [W]')
    axes[0, col].set_title(f'HVAC Power - {label}')
    axes[0, col].grid(True, linestyle=':', alpha=0.6)

    # [1,col] 累计成本
    axes[1, col].plot(data['time'], data['cum_cost'], color=color, linewidth=2)
    axes[1, col].set_ylabel('Cum. Cost [USD]')
    axes[1, col].set_title(f'Cumulative Cost - {label}')
    axes[1, col].grid(True, linestyle=':', alpha=0.6)

    # [2,col] TES SOC
    axes[2, col].plot(data['time'], data['ySOCtes'], color=color, linewidth=1.5, label='SOC')
    axes[2, col].fill_between(data['time'], data['ySOCtes'], alpha=0.3, color=color)
    axes[2, col].set_ylabel('SOC [-]')
    axes[2, col].set_ylim(0, 1)
    axes[2, col].set_title(f'TES State - {label}')
    axes[2, col].grid(True, linestyle=':', alpha=0.6)

    # [3,col] SigTES 与电价
    ax_sig = axes[3, col]
    ax_sig.step(data['time'], data['sig_tes'], color=color, alpha=0.7, where='post', linewidth=2, label='SigTES')
    ax_price = ax_sig.twinx()
    ax_price.step(data['time'], data['price'], color='black', alpha=0.4, where='post', linestyle='--', label='Price')
    ax_sig.set_ylabel('SigTES')
    ax_price.set_ylabel('Price [USD/kWh]')
    ax_sig.set_xlabel('Time [Days]')
    ax_sig.set_title(f'Control & Price - {label}')
    ax_sig.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()
