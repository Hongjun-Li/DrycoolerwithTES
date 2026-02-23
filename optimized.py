import numpy as np
import matplotlib.pyplot as plt
from pyfmi import load_fmu
from tqdm import tqdm

# 1. 加载 FMU
fmu_path = 'ASHRAE26_ChillerPlant_0tes_DataCenterDryFMU.fmu'

# 仿真参数（从第191天开始，模拟2天）
SIM_START_DAY = 1
SIM_DAYS = 7
start_time = 86400.0 * (SIM_START_DAY - 1)
stop_time = start_time + 86400.0 * SIM_DAYS
step_size = 300.0             # 5分钟步长（兼顾速度与分辨率）
steps = int((stop_time - start_time) / step_size)

# MPC 参数（强化套利）
MPC_HORIZON_STEPS = 24        # 24 * 5min = 120min 预测窗
SOC_MIN = 0.05
SOC_MAX = 0.95
SWITCH_PENALTY = 0.03         # 控制动作切换惩罚（元）
CHARGE_SETPOINT_DROP_K = 5.0  # 与 baseline.py 一致：充电且夜间时降5°C
LOW_PRICE_THRESHOLD = 0.5
HIGH_PRICE_THRESHOLD = 1.2
CHARGE_NO_GAIN_MAX_STEPS = 3  # 连续3步SOC不增长则退出充电
SOC_RISE_EPS = 1e-4
CHARGE_PRICE_MAX = 0.8        # 允许充电的最高电价（谷+平价）
TERMINAL_SOC_TARGET = 0.55    # 终端SOC目标（提升后续峰价可放电能力）
TERMINAL_SOC_WEIGHT = 0.35    # 终端SOC价值权重
PEAK_DISCHARGE_REWARD = 0.08  # 峰价放电奖励（每步）

# 定义变量列表
plot_vars = ['yPHVAC', 'yPIT', 'yPDCTFan', 'yPpum', 'yTdry', 'yTCWLeaTow', 'yTCDUSup', 'yTCDURet', 'ySOCtes']

def get_electricity_price(time_seconds):
    """
    定义电价曲线 (单位: 元/kWh)
    00:00 - 08:00: 0.5 (谷)
    08:00 - 12:00: 1.2 (峰)
    12:00 - 17:00: 0.8 (平)
    17:00 - 21:00: 1.2 (峰)
    21:00 - 24:00: 0.5 (谷)
    """
    hour = (time_seconds % 86400) / 3600
    if (8 <= hour < 12) or (17 <= hour < 21):
        return 1.2
    elif 12 <= hour < 17:
        return 0.8
    else:
        return 0.5


def is_low_price(time_seconds):
    return get_electricity_price(time_seconds) <= LOW_PRICE_THRESHOLD


def is_high_price(time_seconds):
    return get_electricity_price(time_seconds) >= HIGH_PRICE_THRESHOLD


def is_charge_allowed(time_seconds):
    return get_electricity_price(time_seconds) <= CHARGE_PRICE_MAX


def init_model(model):
    model.reset()
    model.setup_experiment(start_time=start_time)
    model.enter_initialization_mode()
    model.exit_initialization_mode()


def safe_get_tdry(model):
    try:
        return model.get('weaBus.TDryBul')[0]
    except Exception:
        try:
            return model.get('yTdry')[0]
        except Exception:
            return 293.15


def is_night(time_seconds):
    hour = (time_seconds % 86400) / 3600.0
    return not (6.0 <= hour < 22.0)


def compute_tcw_setpoint(t_db_k, sig_tes, now):
    t_base = min(43 + 273.15, max(35 + 273.15, t_db_k + 5.0))
    if sig_tes == 1.0 and is_night(now):
        return t_base - CHARGE_SETPOINT_DROP_K
    return t_base


def build_candidates(soc, now):
    if soc <= SOC_MIN + 0.01:
        sig_candidates = [0.0, 1.0]
    elif soc >= SOC_MAX - 0.01:
        sig_candidates = [0.0, -1.0]
    else:
        sig_candidates = [-1.0, 0.0, 1.0]

    allow_charge = is_charge_allowed(now)
    candidates = []
    for sig in sig_candidates:
        if sig == 1.0 and not allow_charge:
            continue
        candidates.append(sig)
    return candidates


def switching_cost(prev_sig, new_sig):
    if prev_sig == new_sig:
        return 0.0
    if abs(prev_sig - new_sig) == 2.0:
        return SWITCH_PENALTY * 3.0
    return SWITCH_PENALTY


def rollout_cost(model, now, action_sig, prev_sig):
    sig = action_sig
    # 强化套利目标：HVAC 电费 + 切换惩罚 - 峰价放电价值 - 终端SOC价值
    cost = switching_cost(prev_sig, sig)

    for h in range(MPC_HORIZON_STEPS):
        t_h = now + h * step_size
        t_db_h = safe_get_tdry(model)
        t_set = compute_tcw_setpoint(t_db_h, sig, t_h)
        model.set('SigTES', sig)
        model.set('TCWDry', t_set)
        model.do_step(t_h, step_size, True)

        phvac = model.get('yPHVAC')[0]
        soc = model.get('ySOCtes')[0]
        price = get_electricity_price(t_h)

        energy_kwh = phvac * (step_size / 3600000.0)
        cost += energy_kwh * price

        if sig == -1.0 and is_high_price(t_h) and soc > SOC_MIN + 0.03:
            cost -= PEAK_DISCHARGE_REWARD

        if sig == 1.0 and price > CHARGE_PRICE_MAX:
            cost += 10.0

    soc_end = model.get('ySOCtes')[0]
    cost -= TERMINAL_SOC_WEIGHT * max(0.0, soc_end - TERMINAL_SOC_TARGET)

    return cost


def choose_mpc_action(model, now, prev_sig):
    soc = model.get('ySOCtes')[0]
    candidates = build_candidates(soc, now)

    base_state = model.get_fmu_state()
    best_action = candidates[0]
    best_cost = np.inf

    for action in candidates:
        model.set_fmu_state(base_state)
        trial_cost = rollout_cost(model, now, action, prev_sig)
        if trial_cost < best_cost:
            best_cost = trial_cost
            best_action = action

    model.set_fmu_state(base_state)
    model.free_fmu_state(base_state)
    return best_action


def run_baseline(model):
    init_model(model)

    data = {var: [] for var in plot_vars}
    data.update({
        'time': [], 'sig_tes': [], 't_dry_bul': [],
        'tcw_dry_set': [], 'pue': [], 'price': [], 'cum_cost': []
    })

    current_time = start_time
    total_cost = 0.0

    for _ in tqdm(range(steps), desc='仿真(基准模式)'):
        t_db = safe_get_tdry(model)
        sig_tes = 0.0
        t_set = min(43 + 273.15, max(35 + 273.15, t_db + 5.0))

        model.set('SigTES', sig_tes)
        model.set('TCWDry', t_set)
        model.do_step(current_time, step_size, True)

        c_pit = model.get('yPIT')[0]
        c_phvac = model.get('yPHVAC')[0]
        price = get_electricity_price(current_time)
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


def run_mpc(model):
    init_model(model)

    data = {var: [] for var in plot_vars}
    data.update({
        'time': [], 'sig_tes': [], 't_dry_bul': [], 
        'tcw_dry_set': [], 'pue': [], 'price': [], 'cum_cost': []
    })

    current_time = start_time
    sig_tes = 0.0
    total_cost = 0.0
    charge_no_gain_steps = 0
    prev_soc = model.get('ySOCtes')[0]
    charge_soc_gain_acc = 0.0
    charge_steps = 0

    for _ in tqdm(range(steps), desc='仿真(MPC优化)'):
        t_db = safe_get_tdry(model)
        soc_now = model.get('ySOCtes')[0]

        if sig_tes == 1.0:
            if soc_now <= prev_soc + SOC_RISE_EPS:
                charge_no_gain_steps += 1
            else:
                charge_no_gain_steps = 0
            if charge_no_gain_steps >= CHARGE_NO_GAIN_MAX_STEPS:
                sig_tes = 0.0
                charge_no_gain_steps = 0

        if sig_tes == 1.0 and not is_low_price(current_time):
            sig_tes = 0.0
            charge_no_gain_steps = 0

        if not (sig_tes == 1.0 and soc_now < SOC_MAX):
            sig_tes = choose_mpc_action(model, current_time, sig_tes)

        t_set = compute_tcw_setpoint(t_db, sig_tes, current_time)

        model.set('SigTES', sig_tes)
        model.set('TCWDry', t_set)
        model.do_step(current_time, step_size, True)

        c_pit = model.get('yPIT')[0]
        c_phvac = model.get('yPHVAC')[0]
        price = get_electricity_price(current_time)

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

        if sig_tes == 1.0:
            charge_soc_gain_acc += (soc_now - prev_soc)
            charge_steps += 1

        prev_soc = soc_now
        current_time += step_size

    if charge_steps > 0:
        avg_gain = charge_soc_gain_acc / charge_steps
        print(f"[MPC诊断] 充电步数={charge_steps}, 平均每步SOC增量={avg_gain:.6f}")
    else:
        print("[MPC诊断] 本次仿真未进入充电步。")

    return data


# 运行两次仿真对比
model_base = load_fmu(fmu_path)
model_mpc = load_fmu(fmu_path)
res_base = run_baseline(model_base)
res_ctrl = run_mpc(model_mpc)

# 计算节约金额
final_saving = res_base['cum_cost'][-1] - res_ctrl['cum_cost'][-1]
saving_percent = (final_saving / res_base['cum_cost'][-1]) * 100
daily_saving = final_saving / SIM_DAYS

print(f"\n--- HVAC电费经济效益分析 ---")
print(f"基准总HVAC电费 (Baseline): {res_base['cum_cost'][-1]:.2f} 元")
print(f"MPC总HVAC电费 (MPC): {res_ctrl['cum_cost'][-1]:.2f} 元")
print(f"HVAC节省总金额 (Total Saving): {final_saving:.2f} 元 ({saving_percent:.2f}%)")
if final_saving >= 0:
    print(f"经济效果提示：MPC 方案省钱，日均节省约 {daily_saving:.2f} 元/天。")
else:
    print(f"经济效果提示：MPC 方案未省钱，日均多花约 {-daily_saving:.2f} 元/天。")

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
axes[1,0].plot(res_ctrl['time'], res_ctrl['pue'], color='darkred', label='Controlled PUE')
axes[1,0].set_ylim(1.0, 1.2)
axes[1,0].set_ylabel('PUE [-]')
axes[1,0].set_title('Power Usage Effectiveness')

# [2,0] 累计HVAC电费支出
axes[2,0].plot(res_ctrl['time'], res_ctrl['cum_cost'], label='Controlled Cost', color='green')
axes[2,0].plot(res_base['time'], res_base['cum_cost'], label='Baseline Cost', color='gray', linestyle='--')
axes[2,0].set_ylabel('Cum. HVAC Cost [Yuan]')
axes[2,0].legend(loc='upper left')
axes[2,0].set_title('Cumulative HVAC Electricity Cost')

# [3,0] 累计节约金额曲线
saving_curve = np.array(res_base['cum_cost']) - np.array(res_ctrl['cum_cost'])
axes[3,0].fill_between(res_ctrl['time'], saving_curve, color='gold', alpha=0.3, label='Total Saving')
axes[3,0].plot(res_ctrl['time'], saving_curve, color='orange', linewidth=2)
axes[3,0].set_ylabel('Savings [Yuan]')
axes[3,0].set_xlabel('Time [Days]')
axes[3,0].set_title(f'Total Price Saving: {final_saving:.2f} Yuan')

# [0,1] 温度与电价曲线 (右侧 Y 轴)
axes[0,1].plot(res_ctrl['time'], res_ctrl['t_dry_bul'], label='Outdoor Temp', color='orange', alpha=0.4)
ax_price = axes[0,1].twinx()
ax_price.step(res_ctrl['time'], res_ctrl['price'], where='post', color='black', label='Electricity Price')
ax_price.set_ylabel('Price [Yuan/kWh]')
axes[0,1].set_ylabel('Temp [°C]')
axes[0,1].set_title('Environment & Electricity Price')

# [1,1] 水路侧温度
axes[1,1].plot(res_ctrl['time'], np.array(res_ctrl['yTCDUSup'])-273.15, label='T_CDU_Sup')
axes[1,1].plot(res_ctrl['time'], np.array(res_ctrl['yTCDURet'])-273.15, label='T_CDU_Ret')
axes[1,1].set_ylabel('Temp [°C]')
axes[1,1].legend(loc='upper right', fontsize='x-small')
axes[1,1].set_title('Water Loop Temperatures')

# [2,1] 组件功率对比
axes[2,1].plot(res_ctrl['time'], res_ctrl['yPDCTFan'], label='Fan (Ctrl)', color='#ff9999')
axes[2,1].plot(res_base['time'], res_base['yPDCTFan'], label='Fan (Base)', color='#ff9999', linestyle='--', alpha=0.5)
axes[2,1].plot(res_ctrl['time'], res_ctrl['yPpum'], label='Pump (Ctrl)', color='#66b3ff')
axes[2,1].plot(res_base['time'], res_base['yPpum'], label='Pump (Base)', color='#66b3ff', linestyle='--', alpha=0.5)
axes[2,1].set_ylabel('Power [W]')
axes[2,1].legend(loc='upper right', fontsize='x-small', ncol=2)
axes[2,1].set_title('Component Power Analysis')

# [3,1] TES 状态
axes[3,1].plot(res_ctrl['time'], res_ctrl['ySOCtes'], color='green', label='SOC')
ax_sig = axes[3,1].twinx()
ax_sig.step(res_ctrl['time'], res_ctrl['sig_tes'], color='purple', alpha=0.3, where='post')
axes[3,1].set_ylabel('SOC [-]')
ax_sig.set_ylabel('SigTES')
axes[3,1].set_xlabel('Time [Days]')
axes[3,1].set_title('TES Battery Operation')

for ax in axes.flat: ax.grid(True, linestyle=':', alpha=0.6)
plt.show()