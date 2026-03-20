import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime, timedelta

try:
    from pyfmi import load_fmu
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'pyfmi'. Please install or activate an environment "
        "that includes pyfmi before running test_revision.py."
    ) from exc

fmu_path = 'ASHRAE26_ChillerPlant_0tes_DataCenterDryIBMTESFMU.fmu'
model = load_fmu(fmu_path)

SIM_START_DAY = 181
SIM_END_DAY = 188
start_time = 86400.0 * (SIM_START_DAY - 1)
stop_time = 86400.0 * SIM_END_DAY
step_size = 300.0
steps = int((stop_time - start_time) / step_size)

FMU_INPUTS = ['TCWDry', 'SigTES']
FMU_OUTPUTS = [
    'yPHVAC', 'yPIT', 'yPDCTFan', 'yPCWpum', 'yPCDUpum', 'yPTranpum',
    'yPUE', 'yTdry', 'yTCWLeaTow', 'yTCWEntTow', 'yTCDUSup', 'yTCDURet',
    'ySOCtes', 'yQflow', 'yTtesRet'
]

TRANSITION_DELAY = 600.0
MIN_ACTIVE_TIME = 3600.0
MIN_IDLE_TIME = 1800.0
QFLOW_FILTER_ALPHA = 0.2
CHARGE_QFLOW_KEEP_FRAC = 0.10
CHARGE_QFLOW_MIN_ABS = 10.0
LOW_Q_EXIT_DELAY = 900.0
CHARGE_EVAL_DELAY = 900.0
MIN_SOC_GAIN_FOR_CHARGE = 0.01
CHARGE_COOLDOWN = 2700.0


def is_summer(dt):
    return dt.month in [6, 7, 8, 9]


def is_weekday(dt):
    return dt.weekday() < 5


def get_tou_period(time_seconds, base_date):
    dt = base_date + timedelta(seconds=time_seconds)
    hour = dt.hour + dt.minute / 60.0

    if is_summer(dt):
        if is_weekday(dt) and 8 <= hour < 18:
            return "on"
        if is_weekday(dt) and 18 <= hour < 22:
            return "mid"
        return "off"

    if is_weekday(dt) and 8 <= hour < 22:
        return "mid"
    return "off"


def get_electricity_price(time_seconds, base_date):
    dt = base_date + timedelta(seconds=time_seconds)
    hour = dt.hour + dt.minute / 60.0

    if is_summer(dt):
        rate = 26.20
        if is_weekday(dt):
            if 8 <= hour < 22:
                rate += 27.40
            if 8 <= hour < 18:
                rate += 12.77
        return rate / 100.0

    rate = 7.51
    if is_weekday(dt) and 8 <= hour < 22:
        rate += 17.74
    return rate / 100.0


def update_tes_signal(
    mode,
    mode_timer,
    soc,
    q_flow,
    q_flow_scale,
    current_time,
    step,
    base_date,
    charge_low_q_timer,
    charge_cooldown_timer,
    charge_entry_soc,
):
    dt = base_date + timedelta(seconds=current_time)
    current_period = get_tou_period(current_time, base_date)
    current_price = get_electricity_price(current_time, base_date)
    charge_cooldown_timer = max(0.0, charge_cooldown_timer - step)

    future_times = current_time + np.arange(1, 25) * 3600.0
    future_prices = np.array([get_electricity_price(t, base_date) for t in future_times])
    future_periods = [get_tou_period(t, base_date) for t in future_times]

    window_min_price = min(current_price, float(np.min(future_prices)))
    window_max_price = max(current_price, float(np.max(future_prices)))
    price_spread = window_max_price - window_min_price

    charge_start_soc = 0.55
    charge_stop_soc = 0.85
    discharge_start_soc = 0.60
    discharge_stop_soc = 0.25

    near_min_price = current_price <= window_min_price + 0.01
    near_max_price = current_price >= window_max_price - 0.01
    has_high_price_ahead = any(period == "on" for period in future_periods[:18]) or any(
        price >= window_max_price - 0.01 for price in future_prices[:18]
    )

    charge_request = (
        near_min_price
        and soc < charge_start_soc
        and has_high_price_ahead
        and price_spread >= 0.05
        and current_period != "on"
        and charge_cooldown_timer <= 0.0
    )

    discharge_request = (
        near_max_price
        and soc > discharge_start_soc
        and price_spread >= 0.05
    )

    if mode == 'charge':
        active_time = mode_timer + step
        qflow_keep_threshold = max(CHARGE_QFLOW_MIN_ABS, q_flow_scale * CHARGE_QFLOW_KEEP_FRAC)
        if q_flow < qflow_keep_threshold:
            charge_low_q_timer += step
        else:
            charge_low_q_timer = 0.0

        inefficient_charge = (
            active_time >= CHARGE_EVAL_DELAY
            and charge_entry_soc is not None
            and (soc - charge_entry_soc) < MIN_SOC_GAIN_FOR_CHARGE
        )
        low_q_exit = charge_low_q_timer >= LOW_Q_EXIT_DELAY
        hard_stop = soc >= charge_stop_soc or low_q_exit or inefficient_charge
        soft_stop = (not near_min_price) or (not has_high_price_ahead) or (current_period == "on")
        if hard_stop or (active_time >= MIN_ACTIVE_TIME and soft_stop):
            cooldown = CHARGE_COOLDOWN if (low_q_exit or inefficient_charge) else charge_cooldown_timer
            return 'off', 0.0, 0.0, 0.0, cooldown, None
        return 'charge', active_time, 1.0, charge_low_q_timer, charge_cooldown_timer, charge_entry_soc

    if mode == 'discharge':
        active_time = mode_timer + step
        hard_stop = soc <= discharge_stop_soc
        soft_stop = not near_max_price
        if hard_stop or (active_time >= MIN_ACTIVE_TIME and soft_stop):
            return 'off', 0.0, 0.0, 0.0, charge_cooldown_timer, None
        return 'discharge', active_time, -1.0, 0.0, charge_cooldown_timer, None

    if mode == 'charge_pending':
        if charge_request:
            next_timer = mode_timer + step
            if next_timer >= TRANSITION_DELAY:
                return 'charge', 0.0, 1.0, 0.0, charge_cooldown_timer, soc
            return 'charge_pending', next_timer, 0.0, 0.0, charge_cooldown_timer, None
        return 'off', 0.0, 0.0, 0.0, charge_cooldown_timer, None

    if mode == 'discharge_pending':
        if discharge_request:
            next_timer = mode_timer + step
            if next_timer >= TRANSITION_DELAY:
                return 'discharge', 0.0, -1.0, 0.0, charge_cooldown_timer, None
            return 'discharge_pending', next_timer, 0.0, 0.0, charge_cooldown_timer, None
        return 'off', 0.0, 0.0, 0.0, charge_cooldown_timer, None

    idle_time = mode_timer + step
    if idle_time < MIN_IDLE_TIME:
        return 'off', idle_time, 0.0, 0.0, charge_cooldown_timer, None

    if charge_request:
        if step >= TRANSITION_DELAY:
            return 'charge', 0.0, 1.0, 0.0, charge_cooldown_timer, soc
        return 'charge_pending', step, 0.0, 0.0, charge_cooldown_timer, None

    if discharge_request:
        if step >= TRANSITION_DELAY:
            return 'discharge', 0.0, -1.0, 0.0, charge_cooldown_timer, None
        return 'discharge_pending', step, 0.0, 0.0, charge_cooldown_timer, None

    return 'off', idle_time, 0.0, 0.0, charge_cooldown_timer, None


def run_simulation(use_control=True):
    model.reset()
    model.setup_experiment(start_time=start_time)
    model.enter_initialization_mode()
    model.exit_initialization_mode()

    data = {var: [] for var in FMU_OUTPUTS}
    data.update({
        'time': [], 'sig_tes': [], 't_dry_bul': [],
        'tcw_dry_set': [], 'pue': [], 'price': [], 'cum_cost': []
    })

    base_date = datetime(2026, 1, 1)
    current_time = start_time
    sig_tes = 0.0
    total_cost = 0.0
    tes_mode = 'off'
    tes_mode_timer = MIN_IDLE_TIME
    q_flow_filtered = 0.0
    q_flow_scale = CHARGE_QFLOW_MIN_ABS
    charge_low_q_timer = 0.0
    charge_cooldown_timer = 0.0
    charge_entry_soc = None

    desc = "Simulation (Controlled)" if use_control else "Simulation (Baseline)"
    for _ in tqdm(range(steps), desc=desc):
        soc = model.get('ySOCtes')[0]
        q_flow = model.get('yQflow')[0]
        q_flow_filtered = (1.0 - QFLOW_FILTER_ALPHA) * q_flow_filtered + QFLOW_FILTER_ALPHA * q_flow
        q_flow_scale = max(q_flow_scale, abs(q_flow_filtered))

        try:
            t_db = model.get('weaBus.TDryBul')[0]
        except Exception:
            t_db = 293.15

        if use_control:
            tes_mode, tes_mode_timer, sig_tes, charge_low_q_timer, charge_cooldown_timer, charge_entry_soc = update_tes_signal(
                tes_mode,
                tes_mode_timer,
                soc,
                q_flow_filtered,
                q_flow_scale,
                current_time,
                step_size,
                base_date,
                charge_low_q_timer,
                charge_cooldown_timer,
                charge_entry_soc,
            )
        else:
            sig_tes = 0.0
            tes_mode = 'off'
            tes_mode_timer = MIN_IDLE_TIME
            charge_low_q_timer = 0.0
            charge_cooldown_timer = 0.0
            charge_entry_soc = None

        t_set = min(41 + 273.15, max(273.15 + 27, t_db + 5.0))

        model.set('SigTES', sig_tes)
        model.set('TCWDry', t_set)
        model.do_step(current_time, step_size, True)

        c_phvac = model.get('yPHVAC')[0]
        c_pit = model.get('yPIT')[0]
        y_pue = model.get('yPUE')[0]
        price = get_electricity_price(current_time, base_date)

        energy_step_kwh = c_phvac * (step_size / 3600000.0)
        total_cost += energy_step_kwh * price

        data['time'].append(current_time / 86400)
        data['sig_tes'].append(sig_tes)
        data['t_dry_bul'].append(t_db - 273.15)
        data['tcw_dry_set'].append(t_set - 273.15)
        data['pue'].append(y_pue if np.isfinite(y_pue) else ((c_pit + c_phvac) / c_pit if c_pit > 0 else 1.0))
        data['price'].append(price)
        data['cum_cost'].append(total_cost)
        for var in FMU_OUTPUTS:
            data[var].append(model.get(var)[0])

        current_time += step_size

    return data


res_base = run_simulation(use_control=False)
res_ctrl = run_simulation(use_control=True)

final_saving = res_base['cum_cost'][-1] - res_ctrl['cum_cost'][-1]
saving_percent = (final_saving / res_base['cum_cost'][-1]) * 100 if res_base['cum_cost'][-1] != 0 else 0.0

print("\n--- HVAC Electricity Cost Analysis ---")
print(f"Baseline HVAC Cost: {res_base['cum_cost'][-1]:.2f} USD")
print(f"Controlled HVAC Cost: {res_ctrl['cum_cost'][-1]:.2f} USD")
print(f"Total HVAC Saving: {final_saving:.2f} USD ({saving_percent:.2f}%)")

fig, axes = plt.subplots(4, 2, figsize=(12, 7), sharex=True)
plt.subplots_adjust(hspace=0.35, wspace=0.25)

axes[0, 0].plot(res_ctrl['time'], res_ctrl['yPHVAC'], label='Controlled', color='red')
axes[0, 0].plot(res_base['time'], res_base['yPHVAC'], label='Baseline', color='blue', linestyle='--', alpha=0.6)
axes[0, 0].set_ylabel('HVAC Power [W]')
axes[0, 0].legend(loc='upper right', fontsize='small')
axes[0, 0].set_title('Cooling Power Consumption')

pue_ctrl_mean = np.mean(res_ctrl['pue'])
axes[1, 0].plot(res_ctrl['time'], res_ctrl['pue'], color='darkred', label=f'Controlled PUE (Mean: {pue_ctrl_mean:.3f})')
axes[1, 0].set_ylabel('PUE [-]')
axes[1, 0].legend(loc='upper right', fontsize='small')
axes[1, 0].set_title('Power Usage Effectiveness')

t_ctrl_cdu_sup = np.array(res_ctrl['yTCDUSup']) - 273.15
t_base_cdu_sup = np.array(res_base['yTCDUSup']) - 273.15
ctrl_over_45_h = np.sum(t_ctrl_cdu_sup > 45.0) * step_size / 3600.0
base_over_45_h = np.sum(t_base_cdu_sup > 45.0) * step_size / 3600.0

axes[2, 0].plot(
    res_ctrl['time'],
    t_ctrl_cdu_sup,
    label=f'Controlled (>45C: {ctrl_over_45_h:.2f} h)',
    color='green'
)
axes[2, 0].plot(
    res_base['time'],
    t_base_cdu_sup,
    label=f'Baseline (>45C: {base_over_45_h:.2f} h)',
    color='gray',
    linestyle='--'
)
axes[2, 0].axhline(45.0, color='red', linestyle='--', linewidth=1.0, label='45C Threshold')
axes[2, 0].set_ylabel('T_CDU_Sup [C]')
axes[2, 0].legend(loc='lower right', fontsize='x-small')
axes[2, 0].set_title('T_CDU_Sup Curve & Time Above 45C')

saving_curve = np.array(res_base['cum_cost']) - np.array(res_ctrl['cum_cost'])
axes[3, 0].fill_between(res_ctrl['time'], saving_curve, color='gold', alpha=0.3, label='Total Saving')
axes[3, 0].plot(res_ctrl['time'], saving_curve, color='orange', linewidth=2)
axes[3, 0].set_ylabel('Savings [USD]')
axes[3, 0].set_xlabel('Time [Days]')
axes[3, 0].set_title(f'Total Price Saving: {final_saving:.2f} USD')

axes[0, 1].plot(res_ctrl['time'], res_ctrl['t_dry_bul'], label='Outdoor Temp', color='orange', alpha=0.4)
ax_price = axes[0, 1].twinx()
ax_price.step(res_ctrl['time'], res_ctrl['price'], where='post', color='black', label='Electricity Price')
ax_price.set_ylabel('Price [USD/kWh]')
axes[0, 1].set_ylabel('Temp [C]')
axes[0, 1].set_title('Environment & Electricity Price')

axes[1, 1].plot(res_ctrl['time'], np.array(res_ctrl['yTCDUSup']) - 273.15, label='T_CDU_Sup', color='blue', linewidth=1.5)
axes[1, 1].plot(res_ctrl['time'], np.array(res_ctrl['yTCDURet']) - 273.15, label='T_CDU_Ret', color='purple', linewidth=1.5)
axes[1, 1].plot(res_ctrl['time'], np.array(res_ctrl['yTCWLeaTow']) - 273.15, label='T_CW_Lea_Tow', color='orange', linewidth=1.5)
axes[1, 1].plot(res_ctrl['time'], np.array(res_ctrl['yTCWEntTow']) - 273.15, label='T_CW_Ent_Tow', color='red', linewidth=1.5)
axes[1, 1].plot(res_base['time'], np.array(res_base['yTCDUSup']) - 273.15, label='T_CDU_Sup (Base)', color='blue', linestyle='--', alpha=0.5)
axes[1, 1].plot(res_base['time'], np.array(res_base['yTCDURet']) - 273.15, label='T_CDU_Ret (Base)', color='purple', linestyle='--', alpha=0.5)
axes[1, 1].plot(res_base['time'], np.array(res_base['yTCWLeaTow']) - 273.15, label='T_CW_Lea_Tow (Base)', color='orange', linestyle='--', alpha=0.5)
axes[1, 1].plot(res_base['time'], np.array(res_base['yTCWEntTow']) - 273.15, label='T_CW_Ent_Tow (Base)', color='red', linestyle='--', alpha=0.5)
axes[1, 1].set_ylabel('Temp [C]')
axes[1, 1].legend(loc='upper right', fontsize='x-small', ncol=2)
axes[1, 1].set_title('Water Loop Temperatures')

axes[2, 1].plot(res_ctrl['time'], res_ctrl['yPDCTFan'], label='DCT Fan (Ctrl)', color='#ff9999')
axes[2, 1].plot(res_base['time'], res_base['yPDCTFan'], label='DCT Fan (Base)', color='#ff9999', linestyle='--', alpha=0.5)
axes[2, 1].plot(res_ctrl['time'], res_ctrl['yPCWpum'], label='CW Pump (Ctrl)', color='#66b3ff')
axes[2, 1].plot(res_base['time'], res_base['yPCWpum'], label='CW Pump (Base)', color='#66b3ff', linestyle='--', alpha=0.5)
axes[2, 1].plot(res_ctrl['time'], res_ctrl['yPCDUpum'], label='CDU Pump (Ctrl)', color='#4daf4a')
axes[2, 1].plot(res_base['time'], res_base['yPCDUpum'], label='CDU Pump (Base)', color='#4daf4a', linestyle='--', alpha=0.5)
axes[2, 1].plot(res_ctrl['time'], res_ctrl['yPTranpum'], label='Tran Pump (Ctrl)', color='#984ea3')
axes[2, 1].plot(res_base['time'], res_base['yPTranpum'], label='Tran Pump (Base)', color='#984ea3', linestyle='--', alpha=0.5)
axes[2, 1].set_ylabel('Power [W]')
axes[2, 1].legend(loc='upper right', fontsize='x-small', ncol=2)
axes[2, 1].set_title('Component Power Analysis')

axes[3, 1].plot(res_ctrl['time'], res_ctrl['ySOCtes'], color='green', label='SOC')
ax_sig = axes[3, 1].twinx()
ax_sig.step(res_ctrl['time'], res_ctrl['sig_tes'], color='purple', alpha=0.3, where='post')
axes[3, 1].set_ylabel('SOC [-]')
ax_sig.set_ylabel('SigTES')
axes[3, 1].set_xlabel('Time [Days]')
axes[3, 1].set_title('TES Battery Operation')

for ax in axes.flat:
    ax.grid(True, linestyle=':', alpha=0.6)

plt.show()
