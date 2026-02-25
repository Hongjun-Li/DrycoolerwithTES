import numpy as np
import matplotlib.pyplot as plt
from pyfmi import load_fmu
from tqdm import tqdm
from datetime import datetime, timedelta

# 1. 加载 FMU
fmu_path = 'ASHRAE26_ChillerPlant_0tes_DataCenterDryFMU.fmu'
model = load_fmu(fmu_path)

# 仿真参数（第1天到第365天）
SIM_START_DAY = 181
SIM_END_DAY = 188
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


def run_simulation(use_control=True):
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

    soc_charge_stop = 0.85
    soc_discharge_stop = 0.25

    desc = "仿真(受控模式)" if use_control else "仿真(基准模式)"
    for _ in tqdm(range(steps), desc=desc):
        # --- A. 获取传感器读数 ---
        soc = model.get('ySOCtes')[0]
        t_cdu_sup = model.get('yTCDUSup')[0]  # 获取当前CDU供水温度
        try:
            t_db = model.get('weaBus.TDryBul')[0]
        except:
            t_db = 293.15

        if use_control:
            # --- B. TES 控制逻辑：温度优先，达到45°C时强制放电降温 ---
            period = get_tou_period(current_time, base_date)
            dt = base_date + timedelta(seconds=current_time)
            is_summer = 6 <= dt.month <= 9

            # 温度监控：yTCDUSup 接近45°C时强制放电降温
            t_cdu_sup_c = t_cdu_sup - 273.15  # 转换为摄氏度
            temp_critical = t_cdu_sup_c >= 45.0  # 温度达到上限，紧急放电
            temp_high = t_cdu_sup_c >= 44.0  # 温度接近上限，开始放电
            temp_safe = t_cdu_sup_c < 42.0  # 温度恢复安全，可以恢复正常控制

            charge_allowed = period == "off"
            discharge_allowed = (period == "on") if is_summer else (period == "mid")

            # 温度安全优先：温度过高时强制TES放电提供额外冷量
            if temp_critical and soc > soc_discharge_stop:
                # 温度≥45°C：强制放电（除非SOC已耗尽）
                sig_tes = -1.0
            elif temp_high and soc > soc_discharge_stop:
                # 温度≥44°C：强制放电降温
                sig_tes = -1.0
            elif 42.0 <= t_cdu_sup_c < 44.0:
                # 温度在42-44°C之间：禁止充电，如果在充电则停止
                if sig_tes == 1.0:
                    sig_tes = 0.0
                # 如果在放电，检查是否需要停止
                elif sig_tes == -1.0:
                    if soc <= soc_discharge_stop:
                        sig_tes = 0.0
                # 否则保持当前状态
            else:
                # 温度<42°C：恢复基于电价的正常控制逻辑
                if sig_tes == 1.0:  # 充电中
                    if soc >= soc_charge_stop or not charge_allowed:
                        sig_tes = 0.0
                elif sig_tes == -1.0:  # 放电中
                    if soc <= soc_discharge_stop or not discharge_allowed:
                        sig_tes = 0.0
                else:  # 待机状态，根据电价和SOC决定充放电
                    if charge_allowed and soc < soc_charge_stop:
                        sig_tes = 1.0
                    elif discharge_allowed and soc > soc_discharge_stop:
                        sig_tes = -1.0
        else:
            sig_tes = 0.0

        # --- C. Dry Cooler Setpoint 调节 ---
        t_set = min(45 + 273.15, max(40 + 273.15, t_db + 5.0))

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


# 运行两次仿真对比
res_base = run_simulation(use_control=False)
res_ctrl = run_simulation(use_control=True)

# 计算节约金额
final_saving = res_base['cum_cost'][-1] - res_ctrl['cum_cost'][-1]
saving_percent = (final_saving / res_base['cum_cost'][-1]) * 100

print("\n--- HVAC Electricity Cost Analysis ---")
print(f"Baseline HVAC Cost: {res_base['cum_cost'][-1]:.2f} USD")
print(f"Controlled HVAC Cost: {res_ctrl['cum_cost'][-1]:.2f} USD")
print(f"Total HVAC Saving: {final_saving:.2f} USD ({saving_percent:.2f}%)")

# 4. 绘图：4行 x 2列 布局 (增加电价和成本)
fig, axes = plt.subplots(4, 2, figsize=(12, 7), sharex=True)
plt.subplots_adjust(hspace=0.35, wspace=0.25)

# [0,0] PHVAC 对比
axes[0,0].plot(res_ctrl['time'], res_ctrl['yPHVAC'], label='Controlled', color='red')
axes[0,0].plot(res_base['time'], res_base['yPHVAC'], label='Baseline', color='blue', linestyle='--', alpha=0.6)
axes[0,0].set_ylabel('HVAC Power [W]')
axes[0,0].legend(loc='upper right', fontsize='small')
axes[0,0].set_title('Cooling Power Consumption')

# [1,0] PUE 对比
pue_ctrl_mean = np.mean(res_ctrl['pue'])
axes[1,0].plot(res_ctrl['time'], res_ctrl['pue'], color='darkred', label=f'Controlled PUE (Mean: {pue_ctrl_mean:.3f})')
axes[1,0].set_ylim(0.9, 1.1)
axes[1,0].set_ylabel('PUE [-]')
axes[1,0].legend(loc='upper right', fontsize='small')
axes[1,0].set_title('Power Usage Effectiveness')

# [2,0] 基准/控制超过45°C的时间与温度曲线
t_ctrl_cdu_sup = np.array(res_ctrl['yTCDUSup']) - 273.15
t_base_cdu_sup = np.array(res_base['yTCDUSup']) - 273.15
ctrl_over_45_h = np.sum(t_ctrl_cdu_sup > 45.0) * step_size / 3600.0
base_over_45_h = np.sum(t_base_cdu_sup > 45.0) * step_size / 3600.0

axes[2,0].plot(
    res_ctrl['time'],
    t_ctrl_cdu_sup,
    label=f'Controlled (>45°C: {ctrl_over_45_h:.2f} h)',
    color='green'
)
axes[2,0].plot(
    res_base['time'],
    t_base_cdu_sup,
    label=f'Baseline (>45°C: {base_over_45_h:.2f} h)',
    color='gray',
    linestyle='--'
)
axes[2,0].axhline(45.0, color='red', linestyle='--', linewidth=1.0, label='45°C Threshold')
axes[2,0].set_ylabel('T_CDU_Sup [°C]')
axes[2,0].legend(loc='lower right', fontsize='x-small')
axes[2,0].set_title('T_CDU_Sup Curve & Time Above 45°C')
axes[2,0].set_ylim(35, 50)

# [3,0] 累计节约金额曲线
saving_curve = np.array(res_base['cum_cost']) - np.array(res_ctrl['cum_cost'])
axes[3,0].fill_between(res_ctrl['time'], saving_curve, color='gold', alpha=0.3, label='Total Saving')
axes[3,0].plot(res_ctrl['time'], saving_curve, color='orange', linewidth=2)
axes[3,0].set_ylabel('Savings [USD]')
axes[3,0].set_xlabel('Time [Days]')
axes[3,0].set_title(f'Total Price Saving: {final_saving:.2f} USD')

# [0,1] 温度与电价曲线 (右侧 Y 轴)
axes[0,1].plot(res_ctrl['time'], res_ctrl['t_dry_bul'], label='Outdoor Temp', color='orange', alpha=0.4)
ax_price = axes[0,1].twinx()
ax_price.step(res_ctrl['time'], res_ctrl['price'], where='post', color='black', label='Electricity Price')
ax_price.set_ylabel('Price [USD/kWh]')
axes[0,1].set_ylabel('Temp [°C]')
axes[0,1].set_title('Environment & Electricity Price')

# [1,1] 水路侧温度
axes[1,1].plot(res_ctrl['time'], np.array(res_ctrl['yTCDUSup'])-273.15, label='T_CDU_Sup')
axes[1,1].plot(res_ctrl['time'], np.array(res_ctrl['yTCDURet'])-273.15, label='T_CDU_Ret')
axes[1,1].plot(res_base['time'], np.array(res_base['yTCDUSup'])-273.15, label='T_CDU_Sup (Base)', color='green', linestyle='--', alpha=0.7)
axes[1,1].set_ylabel('Temp [°C]')
axes[1,1].legend(loc='upper right', fontsize='x-small')
axes[1,1].set_title('Water Loop Temperatures')

# [2,1] 组件功率对比
axes[2,1].plot(res_ctrl['time'], res_ctrl['yPDCTFan'], label='Fan (Ctrl)', color='#ff9999')
axes[2,1].plot(res_base['time'], res_base['yPDCTFan'], label='Fan (Base)', color='#ff9999', linestyle='--', alpha=0.5)
axes[2,1].plot(res_ctrl['time'], res_ctrl['yPpum'], label='Pump (Ctrl)', color='#66b3ff')
axes[2,1].plot(res_base['time'], res_base['yPpum'], label='Pump (Base)', color='#66b3ff', linestyle='--', alpha=0.5)
axes[2,1].set_ylabel('Power [W]')
axes[2,1].legend(loc='upper right', fontsize='x-small')
axes[2,1].set_title('Component Power Analysis')

# [3,1] TES 状态
axes[3,1].plot(res_ctrl['time'], res_ctrl['ySOCtes'], color='green', label='SOC')
ax_sig = axes[3,1].twinx()
ax_sig.step(res_ctrl['time'], res_ctrl['sig_tes'], color='purple', alpha=0.3, where='post')
axes[3,1].set_ylabel('SOC [-]')
ax_sig.set_ylabel('SigTES')
axes[3,1].set_xlabel('Time [Days]')
axes[3,1].set_title('TES Battery Operation')

for ax in axes.flat:
    ax.grid(True, linestyle=':', alpha=0.6)
plt.show()