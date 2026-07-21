from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from RackInlet import (
    CHARGE_QFLOW_MIN_ABS,
    CITY_CONFIGS,
    MIN_IDLE_TIME,
    OUTPUT_PATH as RACK_OUTPUT_PATH,
    QFLOW_FILTER_ALPHA,
    SIM_END_DAY,
    SIM_START_DAY,
    START_TIME,
    STEP_SIZE,
    STEPS,
    WARMUP_DAYS,
    apply_cdu_supply_limit,
    resolve_fmu_path,
    update_tes_signal,
)
from RackInlet import datetime, load_fmu


COOLING_POWER_OUTPUT_PATH = RACK_OUTPUT_PATH.with_name("cooling_power_comparison.svg")
EQUIPMENT_POWER_OUTPUT_PATH = RACK_OUTPUT_PATH.with_name("equipment_power_comparison.svg")

POWER_OUTPUTS = [
    "yPHVAC",
    "yPDCTFan",
    "yPCWpum",
]

POWER_LABELS = {
    "yPHVAC": "Cooling",
    "yPDCTFan": "Dry cooler fan",
    "yPCWpum": "CW pump",
}


def safe_get(model, variable_name, default=np.nan):
    try:
        return model.get(variable_name)[0]
    except Exception:
        return default


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

        t_set = 273.15 + 37
        t_set = apply_cdu_supply_limit(t_set, y_tcdu_sup, use_control)
        model.set("SigTES", sig_tes)
        model.set("TCWDry", t_set)
        model.do_step(current_time, STEP_SIZE, True)
        current_time += STEP_SIZE

    current_time = START_TIME
    data = {"time": []}
    data.update({var: [] for var in POWER_OUTPUTS})

    desc = f"Simulating {city_config['label']} ({'TES' if use_control else 'No TES'})"
    for _ in tqdm(range(STEPS), desc=desc):
        soc = model.get("ySOCtes")[0]
        q_flow = model.get("yQflow")[0]
        y_tcdu_sup = model.get("yTCDUSup")[0]
        q_flow_filtered = (1.0 - QFLOW_FILTER_ALPHA) * q_flow_filtered + QFLOW_FILTER_ALPHA * q_flow
        q_flow_scale = max(q_flow_scale, abs(q_flow_filtered))

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

        t_set = 273.15 + 37
        t_set = apply_cdu_supply_limit(t_set, y_tcdu_sup, use_control)
        model.set("SigTES", sig_tes)
        model.set("TCWDry", t_set)
        model.do_step(current_time, STEP_SIZE, True)

        data["time"].append(current_time / 86400.0)
        for var in POWER_OUTPUTS:
            data[var].append(safe_get(model, var))

        current_time += STEP_SIZE

    return data


def finite_values(results, var_name):
    values = []
    for city_results in results.values():
        for scenario_data in city_results.values():
            values.extend(np.asarray(scenario_data[var_name], dtype=float))
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def format_date_ticklabels(day_ticks):
    base_date = datetime(2026, 1, 1)
    return [(base_date + timedelta(days=float(day))).strftime("%b %d") for day in day_ticks]


def draw_power_figure(results, power_outputs, figsize):
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, axes = plt.subplots(len(power_outputs), 2, figsize=figsize, sharex="col", squeeze=False)
    fig.subplots_adjust(hspace=0.32, wspace=0.28, left=0.08, right=0.98, bottom=0.13, top=0.90)

    scenario_styles = {
        "no_tes": {"label": "Without TES", "color": "#0066ff", "linestyle": "-", "alpha": 0.85},
        "tes": {"label": "With TES", "color": "#ff1f1f", "linestyle": "-", "alpha": 0.95},
    }
    x_ticks = np.arange(SIM_START_DAY - 1, SIM_END_DAY + 1, 1)
    x_ticklabels = format_date_ticklabels(x_ticks)

    for row, var_name in enumerate(power_outputs):
        values = finite_values(results, var_name)
        if values.size:
            data_min = np.nanmin(values)
            data_max = np.nanmax(values)
            y_span = max(data_max - data_min, data_max * 0.1, 1.0)
            y_min = max(0.0, data_min - y_span * 0.35)
            y_max = max(0.1, data_max + y_span * 0.08)
        else:
            y_min = 0.0
            y_max = 1.0

        for col, city_name in enumerate(("New York", "Miami")):
            ax = axes[row, col]
            city_results = results[city_name]

            for scenario_key in ("no_tes", "tes"):
                style = scenario_styles[scenario_key]
                ax.plot(
                    city_results[scenario_key]["time"],
                    city_results[scenario_key][var_name],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.4,
                    alpha=style["alpha"],
                    label=style["label"],
                )

            ax.set_title(POWER_LABELS[var_name])
            ax.set_ylabel("Power [W]")
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(SIM_START_DAY - 1, SIM_END_DAY)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticklabels)
            ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.35)

            ax.legend(
                loc="lower left",
                frameon=True,
                facecolor="white",
                edgecolor="none",
                framealpha=0.9,
            )
            if row == len(power_outputs) - 1:
                ax.set_xlabel("Date")

    return fig


def main():
    results = {}
    for city_config in CITY_CONFIGS:
        city_name = city_config["label"]
        results[city_name] = {
            "no_tes": run_city_simulation(city_config, use_control=False),
            "tes": run_city_simulation(city_config, use_control=True),
        }

    cooling_fig = draw_power_figure(results, ["yPHVAC"], figsize=(9.2, 2.6))
    cooling_fig.savefig(COOLING_POWER_OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {COOLING_POWER_OUTPUT_PATH}")

    equipment_fig = draw_power_figure(results, ["yPDCTFan", "yPCWpum"], figsize=(9.2, 4.0))
    equipment_fig.savefig(EQUIPMENT_POWER_OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {EQUIPMENT_POWER_OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
