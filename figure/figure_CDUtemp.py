from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

try:
    from pyfmi import load_fmu
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'pyfmi'. Please install or activate an environment "
        "that includes pyfmi before running figure_CDUtemp.py."
    ) from exc


SIM_START_DAY = 205
SIM_END_DAY = 212
WARMUP_DAYS = 7
STEP_SIZE = 600.0
START_TIME = 86400.0 * (SIM_START_DAY - 1)
STOP_TIME = 86400.0 * SIM_END_DAY
STEPS = int((STOP_TIME - START_TIME) / STEP_SIZE)

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
CDU_SUPPLY_MAX_C = 45.0
CDU_SUPPLY_MAX_K = CDU_SUPPLY_MAX_C + 273.15
CDU_SUPPLY_GUARD_BAND_K = 1.0

MIAMI_TOU_RATES = {
    "on_peak_kwh": 0.26,
    "off_peak_kwh": 0.09,
}

NEW_YORK_TOU_RATES = {
    "summer_on_peak_kwh": 0.2786,
    "summer_off_peak_kwh": 0.0522,
    "winter_on_peak_kwh": 0.1711,
    "winter_off_peak_kwh": 0.0522,
}

FMU_OUTPUTS = ["ySOCtes", "yQflow", "yTCDUSup"]

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = Path(__file__).resolve().parent / "cdu_temp_comparison.png"

CITY_CONFIGS = [
    {
        "label": "New York",
        "region": "NewYork",
        "fmu_name": "ASHRAE26_ChillerPlant_0tes_Revision_DataCenterDryIBMTESRevisionFMUNewYork.fmu",
    },
    {
        "label": "Miami",
        "region": "Miami",
        "fmu_name": "ASHRAE26_ChillerPlant_0tes_Revision_DataCenterDryIBMTESRevisionFMUMiami.fmu",
    },
]


def c_to_f(temp_c):
    return temp_c * 9.0 / 5.0 + 32.0


def format_temp_ticklabels(ticks_c):
    return [f"{tick_c:.1f} / {c_to_f(tick_c):.1f}" for tick_c in ticks_c]


def is_summer(dt, region):
    if region == "Miami":
        return 4 <= dt.month <= 10
    return dt.month in [6, 7, 8, 9]


def is_weekday(dt):
    return dt.weekday() < 5


def get_tou_period(time_seconds, base_date, region):
    dt = base_date + timedelta(seconds=time_seconds)
    hour = dt.hour + dt.minute / 60.0

    if region == "Miami":
        if is_weekday(dt) and 12 <= hour < 21:
            return "on"
        return "off"

    if is_weekday(dt) and 8 <= hour < 22:
        return "on"
    return "off"


def get_electricity_price(time_seconds, base_date, region):
    dt = base_date + timedelta(seconds=time_seconds)
    hour = dt.hour + dt.minute / 60.0

    if region == "Miami":
        if is_weekday(dt) and 12 <= hour < 21:
            return MIAMI_TOU_RATES["on_peak_kwh"]
        return MIAMI_TOU_RATES["off_peak_kwh"]

    if is_weekday(dt) and 8 <= hour < 22:
        if is_summer(dt, region):
            return NEW_YORK_TOU_RATES["summer_on_peak_kwh"]
        return NEW_YORK_TOU_RATES["winter_on_peak_kwh"]

    if is_summer(dt, region):
        return NEW_YORK_TOU_RATES["summer_off_peak_kwh"]
    return NEW_YORK_TOU_RATES["winter_off_peak_kwh"]


def update_tes_signal(
    mode,
    mode_timer,
    soc,
    q_flow,
    q_flow_scale,
    current_time,
    step,
    base_date,
    region,
    charge_low_q_timer,
    charge_cooldown_timer,
    charge_entry_soc,
):
    current_period = get_tou_period(current_time, base_date, region)
    current_price = get_electricity_price(current_time, base_date, region)
    charge_cooldown_timer = max(0.0, charge_cooldown_timer - step)

    future_times = current_time + np.arange(1, 25) * 3600.0
    future_prices = np.array([get_electricity_price(t, base_date, region) for t in future_times])
    future_periods = [get_tou_period(t, base_date, region) for t in future_times]

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
    discharge_request = near_max_price and soc > discharge_start_soc and price_spread >= 0.05

    if mode == "charge":
        active_time = mode_timer + step
        qflow_keep_threshold = max(CHARGE_QFLOW_MIN_ABS, q_flow_scale * CHARGE_QFLOW_KEEP_FRAC)
        charge_low_q_timer = charge_low_q_timer + step if q_flow < qflow_keep_threshold else 0.0

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
            return "off", 0.0, 0.0, 0.0, cooldown, None
        return "charge", active_time, 1.0, charge_low_q_timer, charge_cooldown_timer, charge_entry_soc

    if mode == "discharge":
        active_time = mode_timer + step
        hard_stop = soc <= discharge_stop_soc
        soft_stop = not near_max_price
        if hard_stop or (active_time >= MIN_ACTIVE_TIME and soft_stop):
            return "off", 0.0, 0.0, 0.0, charge_cooldown_timer, None
        return "discharge", active_time, -1.0, 0.0, charge_cooldown_timer, None

    if mode == "charge_pending":
        if charge_request:
            next_timer = mode_timer + step
            if next_timer >= TRANSITION_DELAY:
                return "charge", 0.0, 1.0, 0.0, charge_cooldown_timer, soc
            return "charge_pending", next_timer, 0.0, 0.0, charge_cooldown_timer, None
        return "off", 0.0, 0.0, 0.0, charge_cooldown_timer, None

    if mode == "discharge_pending":
        if discharge_request:
            next_timer = mode_timer + step
            if next_timer >= TRANSITION_DELAY:
                return "discharge", 0.0, -1.0, 0.0, charge_cooldown_timer, None
            return "discharge_pending", next_timer, 0.0, 0.0, charge_cooldown_timer, None
        return "off", 0.0, 0.0, 0.0, charge_cooldown_timer, None

    idle_time = mode_timer + step
    if idle_time < MIN_IDLE_TIME:
        return "off", idle_time, 0.0, 0.0, charge_cooldown_timer, None

    if charge_request:
        if step >= TRANSITION_DELAY:
            return "charge", 0.0, 1.0, 0.0, charge_cooldown_timer, soc
        return "charge_pending", step, 0.0, 0.0, charge_cooldown_timer, None

    if discharge_request:
        if step >= TRANSITION_DELAY:
            return "discharge", 0.0, -1.0, 0.0, charge_cooldown_timer, None
        return "discharge_pending", step, 0.0, 0.0, charge_cooldown_timer, None

    return "off", idle_time, 0.0, 0.0, charge_cooldown_timer, None


def apply_cdu_supply_limit(t_set, y_tcdu_sup, use_control):
    if not use_control:
        return t_set

    limited_t_set = min(t_set, CDU_SUPPLY_MAX_K - CDU_SUPPLY_GUARD_BAND_K)
    if y_tcdu_sup >= CDU_SUPPLY_MAX_K:
        limited_t_set = min(limited_t_set, y_tcdu_sup - CDU_SUPPLY_GUARD_BAND_K)
    return limited_t_set


def resolve_fmu_path(fmu_name):
    direct_path = ROOT_DIR / fmu_name
    if direct_path.exists():
        return direct_path

    matches = list(ROOT_DIR.rglob(fmu_name))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"FMU not found: {fmu_name}")


def run_city_simulation(city_config, use_control):
    model = load_fmu(str(resolve_fmu_path(city_config["fmu_name"])))
    warmup_start_time = max(0.0, START_TIME - 86400.0 * WARMUP_DAYS)
    warmup_steps = int((START_TIME - warmup_start_time) / STEP_SIZE)

    model.reset()
    model.setup_experiment(start_time=warmup_start_time)
    model.enter_initialization_mode()
    model.exit_initialization_mode()

    base_date = datetime(2026, 1, 1)
    current_time = warmup_start_time
    tes_mode = "off"
    tes_mode_timer = MIN_IDLE_TIME
    q_flow_filtered = 0.0
    q_flow_scale = CHARGE_QFLOW_MIN_ABS
    charge_low_q_timer = 0.0
    charge_cooldown_timer = 0.0
    charge_entry_soc = None
    sig_tes = 0.0

    for _ in range(warmup_steps):
        soc = model.get("ySOCtes")[0]
        q_flow = model.get("yQflow")[0]
        y_tcdu_sup = model.get("yTCDUSup")[0]
        q_flow_filtered = (1.0 - QFLOW_FILTER_ALPHA) * q_flow_filtered + QFLOW_FILTER_ALPHA * q_flow
        q_flow_scale = max(q_flow_scale, abs(q_flow_filtered))

        try:
            t_db = model.get("weaBus.TDryBul")[0]
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
                STEP_SIZE,
                base_date,
                city_config["region"],
                charge_low_q_timer,
                charge_cooldown_timer,
                charge_entry_soc,
            )
        else:
            sig_tes = 0.0

        t_set = min(41 + 273.15, max(273.15 + 37, t_db + 6.0))
        t_set = apply_cdu_supply_limit(t_set, y_tcdu_sup, use_control)
        model.set("SigTES", sig_tes)
        model.set("TCWDry", t_set)
        model.do_step(current_time, STEP_SIZE, True)
        current_time += STEP_SIZE

    current_time = START_TIME
    data = {var: [] for var in FMU_OUTPUTS}
    data["time"] = []

    desc = f"Simulating {city_config['label']} ({'TES' if use_control else 'No TES'})"
    for _ in tqdm(range(STEPS), desc=desc):
        soc = model.get("ySOCtes")[0]
        q_flow = model.get("yQflow")[0]
        y_tcdu_sup = model.get("yTCDUSup")[0]
        q_flow_filtered = (1.0 - QFLOW_FILTER_ALPHA) * q_flow_filtered + QFLOW_FILTER_ALPHA * q_flow
        q_flow_scale = max(q_flow_scale, abs(q_flow_filtered))

        try:
            t_db = model.get("weaBus.TDryBul")[0]
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
                STEP_SIZE,
                base_date,
                city_config["region"],
                charge_low_q_timer,
                charge_cooldown_timer,
                charge_entry_soc,
            )
        else:
            sig_tes = 0.0

        t_set = min(41 + 273.15, max(273.15 + 27, t_db + 6.0))
        t_set = apply_cdu_supply_limit(t_set, y_tcdu_sup, use_control)
        model.set("SigTES", sig_tes)
        model.set("TCWDry", t_set)
        model.do_step(current_time, STEP_SIZE, True)

        data["time"].append(current_time / 86400.0)
        for var in FMU_OUTPUTS:
            value = model.get(var)[0]
            if var == "yTCDUSup":
                value = value - 273.15
            data[var].append(value)

        current_time += STEP_SIZE

    return data


def draw_figure(results):
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 2.3), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.30, left=0.08, right=0.98, bottom=0.23, top=0.84)

    temp_values = []
    for city_data in results.values():
        for scenario_data in city_data.values():
            temp_values.extend(scenario_data["yTCDUSup"])

    temp_min = min(39.5, np.floor(min(temp_values) * 2.0) / 2.0)
    temp_max = max(45.5, np.ceil(max(temp_values) * 2.0) / 2.0)
    temp_ticks = np.arange(temp_min, temp_max + 0.1, 2.5)

    line_colors = {
        "no_tes": "#666666",
        "tes": "#2ca02c",
    }
    scenario_labels = {
        "no_tes": "Baseline",
        "tes": "Controlled",
    }

    for ax, city_name in zip(axes, ("New York", "Miami")):
        city_results = results[city_name]
        handles = []
        labels = []

        for scenario_key in ("tes", "no_tes"):
            scenario_data = city_results[scenario_key]
            over_45_hours = np.sum(np.array(scenario_data["yTCDUSup"]) > 45.0) * STEP_SIZE / 3600.0
            line = ax.plot(
                scenario_data["time"],
                scenario_data["yTCDUSup"],
                color=line_colors[scenario_key],
                linewidth=1.4,
                alpha=0.95,
                label=f"{scenario_labels[scenario_key]} (>45C: {over_45_hours:.2f} h)",
            )[0]
            handles.append(line)
            labels.append(line.get_label())

        threshold = ax.axhline(
            45.0,
            color="#ff4d4d",
            linestyle="--",
            linewidth=1.1,
            label="45C Threshold",
        )
        handles.append(threshold)
        labels.append("45C Threshold")

        ax.set_title(f"{city_name}\nCDU supply temperature and hours above 45C")
        ax.set_ylabel("T_CDU_Sup [C / F]")
        ax.set_xlim(SIM_START_DAY - 1, SIM_END_DAY)
        ax.set_xticks(np.arange(SIM_START_DAY - 1, SIM_END_DAY + 1, 1))
        ax.set_ylim(temp_min, temp_max)
        ax.set_yticks(temp_ticks)
        ax.set_yticklabels(format_temp_ticklabels(temp_ticks))
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.35)
        if city_name == "New York":
            ax.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.63), frameon=False)
        else:
            ax.legend(handles, labels, loc="lower right", frameon=False)
        ax.set_xlabel("Time [Days]")

    return fig


def main():
    results = {}
    for city_config in CITY_CONFIGS:
        city_name = city_config["label"]
        results[city_name] = {
            "no_tes": run_city_simulation(city_config, use_control=False),
            "tes": run_city_simulation(city_config, use_control=True),
        }

    fig = draw_figure(results)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
