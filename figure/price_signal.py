import numpy as np
import matplotlib.pyplot as plt
from pyfmi import load_fmu
from tqdm import tqdm
from datetime import datetime, timedelta

# 1. 加载 FMU
fmu_path = 'ASHRAE26_ChillerPlant_0tes_DataCenterDryFMU.fmu'
model = load_fmu(fmu_path)

# 仿真参数（第181天到第188天）
SIM_START_DAY = 181
SIM_END_DAY = 188
start_time = 86400.0 * (SIM_START_DAY - 1)
stop_time = 86400.0 * SIM_END_DAY
step_size = 60.0             # 1分钟步长
steps = int((stop_time - start_time) / step_size)

# 定义变量列表
plot_vars = ['yPHVAC', 'yPIT', 'yPDCTFan', 'yPpum', 'yTdry', 'yTCWLeaTow', 'yTCWEntTow', 'yTCDUSup', 'yTCDURet', 'ySOCtes']

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


def forecast_average_price(time_seconds, base_date, horizon_hours=6, sample_seconds=900):
    """Estimate mean TOU price over a future horizon (default 6h, sampled every 15 min)."""
    horizon_seconds = max(sample_seconds, horizon_hours * 3600)
    steps = max(1, int(horizon_seconds / sample_seconds))
    prices = []
    for i in range(1, steps + 1):
        future_time = time_seconds + i * sample_seconds
        prices.append(get_electricity_price(future_time, base_date))
    return float(np.mean(prices)) if prices else get_electricity_price(time_seconds, base_date)


def upcoming_on_peak(time_seconds, base_date, window_hours=3, sample_seconds=900):
    """Check if an on-peak window starts within the next window_hours."""
    future_span = max(1, int((window_hours * 3600) / sample_seconds))
    for i in range(1, future_span + 1):
        future_time = time_seconds + i * sample_seconds
        if get_tou_period(future_time, base_date) == "on":
            return True
    return False


def solve_tes_qp(price, soc, temp_cdu_sup_c, u_prev, soc_gain, soc_ref):
    """
    单步 QP（标量）:
          min_u   w_u*u^2 + w_s*(u-u_prev)^2 + w_p*p_norm*u
              + w_soc*(soc-soc_ref)*u + w_t*max(0, T-44)*u
      s.t.    -1 <= u <= 1
              soc_min <= soc + soc_gain*u <= soc_max
              T_pred = T + alpha_t*u <= 45

    其中 u 为连续 TES 控制信号（充电>0，放电<0），随后映射到 {-1,0,1}。
    """
    soc_min = 0.20
    soc_max = 0.90

    # 经验温度灵敏度：u>0（充电）会抬升温度，u<0（放电）有助于降温
    alpha_t = 0.8

    # 价格标准化（基于当前脚本电价范围）
    p_low = 0.05683
    p_high = 0.22065
    p_mid = 0.5 * (p_low + p_high)
    p_span = max(1e-6, p_high - p_low)
    p_norm = (price - p_mid) / p_span

    # QP 权重（允许正常充放，避免抖振）
    w_u = 0.10
    w_s = 1.60
    w_p = 2.20
    w_soc = 2.80
    w_t = 3.00

    # 二次目标: a*u^2 + b*u
    a = w_u + w_s
    b = (
        (-2.0 * w_s * u_prev)
        + (w_p * p_norm)
        + (w_soc * (soc - soc_ref))
        + (w_t * max(0.0, temp_cdu_sup_c - 44.0))
    )

    # 可行域边界
    lb = -1.0
    ub = 1.0

    eff_soc_gain = soc_gain if soc_gain > 1e-6 else 0.001
    lb_soc = (soc_min - soc) / eff_soc_gain
    ub_soc = (soc_max - soc) / eff_soc_gain
    lb = max(lb, lb_soc)
    ub = min(ub, ub_soc)

    # 价格门控：保留经济性导向，但不“锁死”放电机会
    charge_price_cap = 0.080
    discharge_price_floor = 0.095
    if price >= charge_price_cap:
        ub = min(ub, 0.0)
    if price <= discharge_price_floor:
        lb = max(lb, 0.0)

    ub_temp = (45.0 - temp_cdu_sup_c) / alpha_t
    ub = min(ub, ub_temp)

    if lb > ub:
        u_cont = lb if abs(lb) < abs(ub) else ub
    else:
        u_free = -b / (2.0 * a)
        u_cont = min(ub, max(lb, u_free))

    # 映射到 FMU 常用离散控制信号
    if u_cont > 0.40:
        sig_tes = 1.0
    elif u_cont < -0.40:
        sig_tes = -1.0
    else:
        sig_tes = 0.0

    return u_cont, sig_tes


def run_simulation(use_control=True):
    model.reset()
    model.setup_experiment(start_time=start_time)
    model.enter_initialization_mode()
    model.exit_initialization_mode()

    data = {var: [] for var in plot_vars}
    data.update({
        'time': [], 'sig_tes': [], 't_dry_bul': [],
        'tcw_dry_set': [], 'pue': [], 'price': [], 'cum_cost': [], 'u_tes_qp': []
    })

    base_date = datetime(2026, 1, 1)
    current_time = start_time
    sig_tes = 0.0
    u_tes_prev = 0.0
    total_cost = 0.0
    soc_prev = None
    soc_gain_est = 0.001
    last_sig_tes = 0.0
    hold_timer_steps = 0
    control_interval_steps = int(15 * 60 / step_size)   # 每15分钟决策一次
    min_hold_steps = int(12 * 60 / step_size)           # 最小保持12分钟，减少抖动
    step_idx = 0

    desc = "仿真(受控模式)" if use_control else "仿真(基准模式)"
    for _ in tqdm(range(steps), desc=desc):
        # --- A. 获取传感器读数 ---
        soc = model.get('ySOCtes')[0]
        try:
            t_db = model.get('weaBus.TDryBul')[0]
        except:
            t_db = 293.15
        t_cdu_sup_c = model.get('yTCDUSup')[0] - 273.15

        if use_control:
            # --- B. TES QP 控制逻辑 (省钱 + T_CDU_Sup<=45°C) ---
            price = get_electricity_price(current_time, base_date)
            period = get_tou_period(current_time, base_date)

            # SOC 参考轨迹：低价储能，高价释能
            if period == "off":
                base_soc_ref = 0.78
            elif period == "mid":
                base_soc_ref = 0.38
            else:
                base_soc_ref = 0.30

            future_avg_price = forecast_average_price(current_time, base_date)
            price_delta = future_avg_price - price
            soc_bias = float(np.clip(price_delta / 0.05, -0.25, 0.25))
            soc_ref = float(np.clip(base_soc_ref + soc_bias, 0.25, 0.90))

            if upcoming_on_peak(current_time, base_date, window_hours=3):
                soc_ref = max(soc_ref, 0.85)
            if period == "on" and price_delta < -0.02:
                soc_ref = min(soc_ref, 0.35)

            u_tes_raw, sig_qp = solve_tes_qp(
                price=price,
                soc=soc,
                temp_cdu_sup_c=t_cdu_sup_c,
                u_prev=u_tes_prev,
                soc_gain=soc_gain_est,
                soc_ref=soc_ref
            )

            # 低频调度 + 最小驻留 + 温度安全优先
            force_discharge = (t_cdu_sup_c >= 44.7) and (soc > 0.22)
            decision_tick = (step_idx % max(1, control_interval_steps) == 0)

            if force_discharge:
                sig_tes = -1.0
                u_tes = -1.0
                hold_timer_steps = min_hold_steps
            elif hold_timer_steps > 0:
                sig_tes = last_sig_tes
                u_tes = sig_tes
                hold_timer_steps -= 1
            elif decision_tick:
                sig_tes = sig_qp
                u_tes = u_tes_raw
                if sig_tes != last_sig_tes:
                    hold_timer_steps = min_hold_steps
                last_sig_tes = sig_tes
            else:
                sig_tes = last_sig_tes
                u_tes = u_tes_prev
        else:
            u_tes = 0.0
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
        data['u_tes_qp'].append(u_tes)
        for var in plot_vars:
            data[var].append(model.get(var)[0])

        # 在线更新 SOC 增益估计（用于下一步 QP 约束）
        soc_now = data['ySOCtes'][-1]
        if soc_prev is not None and abs(u_tes_prev) > 0.2:
            gain_candidate = (soc_now - soc_prev) / u_tes_prev
            if np.isfinite(gain_candidate) and gain_candidate > 1e-6:
                soc_gain_est = 0.9 * soc_gain_est + 0.1 * gain_candidate
        soc_prev = soc_now
        u_tes_prev = u_tes

        current_time += step_size
        step_idx += 1
    return data


# 运行两次仿真对比
res_base = run_simulation(use_control=False)
res_ctrl = run_simulation(use_control=True)

# 计算节约金额
final_saving = res_base['cum_cost'][-1] - res_ctrl['cum_cost'][-1]
saving_percent = (final_saving / res_base['cum_cost'][-1]) * 100

# TES 切换频率统计
sig_arr = np.array(res_ctrl['sig_tes'])
switch_count = int(np.sum(np.abs(np.diff(sig_arr)) > 0.5))
sim_hours = (stop_time - start_time) / 3600.0
switch_per_hour = switch_count / sim_hours if sim_hours > 0 else 0.0

print("\n--- HVAC Electricity Cost Analysis ---")
print(f"Baseline HVAC Cost: {res_base['cum_cost'][-1]:.2f} USD")
print(f"Controlled HVAC Cost: {res_ctrl['cum_cost'][-1]:.2f} USD")
print(f"Total HVAC Saving: {final_saving:.2f} USD ({saving_percent:.2f}%)")
print(f"TES Switch Count: {switch_count} ({switch_per_hour:.2f} switches/hour)")

# 4. 绘图：仅展示温度、电价与 TES 运行（三个竖向子图）
fig, axes = plt.subplots(3, 1, figsize=(16, 6), sharex=True)
plt.subplots_adjust(hspace=0.3)

# (1) 环境温度
axes[0].plot(res_ctrl['time'], res_ctrl['t_dry_bul'], color='#f4a261', label='Outdoor Temp')
axes[0].set_ylabel('Temp [°C]')
axes[0].set_title('Outdoor Temperature')
axes[0].legend(loc='upper right', fontsize='small')

# (2) 电价
axes[1].step(res_ctrl['time'], res_ctrl['price'], where='post', color='grey', linewidth=1.5, label='Electricity Price')
axes[1].set_ylabel('Price [USD/kWh]')
axes[1].set_title('Electricity Price')
axes[1].legend(loc='upper right', fontsize='small')

# (3) TES 运行状态
axes[2].plot(res_ctrl['time'], res_ctrl['ySOCtes'], color='green', linewidth=1.5, label='SOC')
ax_sig = axes[2].twinx()
ax_sig.step(res_ctrl['time'], res_ctrl['sig_tes'], color='purple', alpha=0.3, where='post', label='SigTES')
axes[2].set_ylabel('SOC [-]')
ax_sig.set_ylabel('SigTES')
axes[2].set_xlabel('Time [Days]')
axes[2].set_title('TES Operation')
axes[2].legend(loc='upper left', fontsize='small')

for ax in axes:
    ax.grid(True, linestyle=':', alpha=0.6)
plt.show()