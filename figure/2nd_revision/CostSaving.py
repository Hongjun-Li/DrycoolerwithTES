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
    get_electricity_price,
    resolve_fmu_path,
    update_tes_signal,
)
from RackInlet import datetime, load_fmu


OUTPUT_PATH = RACK_OUTPUT_PATH.with_name("energy_cost_saving_comparison.svg")


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
    total_energy_kwh = 0.0
    total_cost_usd = 0.0

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

        hvac_power_w = model.get("yPHVAC")[0]
        energy_step_kwh = hvac_power_w * STEP_SIZE / 3600000.0
        price = get_electricity_price(current_time, base_date, city_config["region"])
        total_energy_kwh += energy_step_kwh
        total_cost_usd += energy_step_kwh * price

        current_time += STEP_SIZE

    return {
        "total_energy_kwh": total_energy_kwh,
        "total_cost_usd": total_cost_usd,
    }


def build_summary(results):
    summary = {}
    for city_name, city_results in results.items():
        baseline = city_results["no_tes"]
        controlled = city_results["tes"]
        energy_saving = baseline["total_energy_kwh"] - controlled["total_energy_kwh"]
        cost_saving = baseline["total_cost_usd"] - controlled["total_cost_usd"]
        summary[city_name] = {
            "baseline_energy_kwh": baseline["total_energy_kwh"],
            "controlled_energy_kwh": controlled["total_energy_kwh"],
            "energy_saving_kwh": energy_saving,
            "energy_saving_percent": energy_saving / baseline["total_energy_kwh"] * 100.0
            if baseline["total_energy_kwh"]
            else 0.0,
            "baseline_cost_usd": baseline["total_cost_usd"],
            "controlled_cost_usd": controlled["total_cost_usd"],
            "cost_saving_usd": cost_saving,
            "cost_saving_percent": cost_saving / baseline["total_cost_usd"] * 100.0
            if baseline["total_cost_usd"]
            else 0.0,
        }
    return summary


def draw_figure(summary):
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    city_display_names = {
        "New York": "Poughkeepsie, NY",
        "Miami": "Miami, FL",
    }
    city_names = list(summary.keys())
    x_ticklabels = [city_display_names.get(city, city) for city in city_names]
    x = np.arange(len(city_names))
    width = 0.20

    baseline_energy_kwh = np.array([summary[city]["baseline_energy_kwh"] for city in city_names])
    controlled_energy_kwh = np.array([summary[city]["controlled_energy_kwh"] for city in city_names])
    energy_saving_percent = np.array([summary[city]["energy_saving_percent"] for city in city_names])
    baseline_cost = np.array([summary[city]["baseline_cost_usd"] for city in city_names])
    controlled_cost = np.array([summary[city]["controlled_cost_usd"] for city in city_names])
    cost_saving_percent = np.array([summary[city]["cost_saving_percent"] for city in city_names])

    baseline_color = "#0066ff"
    controlled_color = "#ff1f1f"

    fig, (ax_energy, ax_cost) = plt.subplots(1, 2, figsize=(9.2, 3.5))
    fig.subplots_adjust(wspace=0.30, left=0.08, right=0.98, bottom=0.18, top=0.84)

    bars_base = ax_energy.bar(
        x - width / 2,
        baseline_energy_kwh,
        width,
        color=baseline_color,
        label="Without TES",
    )
    bars_ctrl = ax_energy.bar(
        x + width / 2,
        controlled_energy_kwh,
        width,
        color=controlled_color,
        label="With TES",
    )
    ax_energy.set_title("Total cooling energy comparison")
    ax_energy.set_ylabel("Cooling energy [kWh]")
    ax_energy.set_xticks(x)
    ax_energy.set_xticklabels(x_ticklabels)
    ax_energy.legend(frameon=False)
    ax_energy.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.35)
    ax_energy.set_ylim(0.0, max(np.max(baseline_energy_kwh), np.max(controlled_energy_kwh)) * 1.18)

    ax_energy.bar_label(bars_base, fmt="%.1f", padding=3, fontsize=9)
    energy_labels = [
        f"{energy:.1f}\n({percent:.2f}%)"
        for energy, percent in zip(controlled_energy_kwh, energy_saving_percent)
    ]
    ax_energy.bar_label(bars_ctrl, labels=energy_labels, padding=3, fontsize=9)

    bars_cost_base = ax_cost.bar(
        x - width / 2,
        baseline_cost,
        width,
        color=baseline_color,
        label="Without TES",
    )
    bars_cost_ctrl = ax_cost.bar(
        x + width / 2,
        controlled_cost,
        width,
        color=controlled_color,
        label="With TES",
    )
    ax_cost.set_title("HVAC electricity cost comparison")
    ax_cost.set_ylabel("Total cost [USD]")
    ax_cost.set_xticks(x)
    ax_cost.set_xticklabels(x_ticklabels)
    ax_cost.legend(frameon=False)
    ax_cost.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.35)
    ax_cost.set_ylim(0.0, max(np.max(baseline_cost), np.max(controlled_cost)) * 1.18)

    ax_cost.bar_label(bars_cost_base, fmt="$%.2f", padding=3, fontsize=9)
    cost_labels = [
        f"${cost:.2f}\n({percent:.2f}%)"
        for cost, percent in zip(controlled_cost, cost_saving_percent)
    ]
    ax_cost.bar_label(bars_cost_ctrl, labels=cost_labels, padding=3, fontsize=9)

    return fig


def print_summary(summary):
    print("\n--- Total Energy and Cost Saving Summary ---")
    for city_name, values in summary.items():
        print(f"{city_name}:")
        print(f"  Baseline energy:   {values['baseline_energy_kwh']:.2f} kWh")
        print(f"  Controlled energy: {values['controlled_energy_kwh']:.2f} kWh")
        print(
            f"  Energy saving:     {values['energy_saving_kwh']:.2f} kWh "
            f"({values['energy_saving_percent']:.2f}%)"
        )
        print(f"  Baseline cost:     {values['baseline_cost_usd']:.2f} USD")
        print(f"  Controlled cost:   {values['controlled_cost_usd']:.2f} USD")
        print(
            f"  Cost saving:       {values['cost_saving_usd']:.2f} USD "
            f"({values['cost_saving_percent']:.2f}%)"
        )


def main():
    results = {}
    for city_config in CITY_CONFIGS:
        city_name = city_config["label"]
        results[city_name] = {
            "no_tes": run_city_simulation(city_config, use_control=False),
            "tes": run_city_simulation(city_config, use_control=True),
        }

    summary = build_summary(results)
    print_summary(summary)

    fig = draw_figure(summary)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
