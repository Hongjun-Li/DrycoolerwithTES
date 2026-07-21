from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from RackInlet import (
    CHARGE_QFLOW_MIN_ABS,
    CITY_CONFIGS,
    FMU_OUTPUTS,
    MIN_IDLE_TIME,
    OUTPUT_PATH as RACK_OUTPUT_PATH,
    QFLOW_FILTER_ALPHA,
    SIM_END_DAY,
    SIM_START_DAY,
    START_TIME,
    STEP_SIZE,
    STEPS,
    STOP_TIME,
    WARMUP_DAYS,
    apply_cdu_supply_limit,
    get_electricity_price,
    resolve_fmu_path,
    update_tes_signal,
)
from RackInlet import datetime, load_fmu


OUTPUT_PATH = RACK_OUTPUT_PATH.with_name("price_behavior.svg")


def run_city_simulation(city_config):
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

        t_set = 273.15 + 37
        t_set = apply_cdu_supply_limit(t_set, y_tcdu_sup, use_control=True)
        model.set("SigTES", sig_tes)
        model.set("TCWDry", t_set)
        model.do_step(current_time, STEP_SIZE, True)
        current_time += STEP_SIZE

    current_time = START_TIME
    data = {var: [] for var in FMU_OUTPUTS}
    data.update({"time": [], "price": [], "t_dry_bul": [], "sig_tes": []})

    for _ in tqdm(range(STEPS), desc=f"Simulating {city_config['label']}"):
        soc = model.get("ySOCtes")[0]
        q_flow = model.get("yQflow")[0]
        y_tcdu_sup = model.get("yTCDUSup")[0]
        q_flow_filtered = (1.0 - QFLOW_FILTER_ALPHA) * q_flow_filtered + QFLOW_FILTER_ALPHA * q_flow
        q_flow_scale = max(q_flow_scale, abs(q_flow_filtered))

        try:
            t_db = model.get("weaBus.TDryBul")[0]
        except Exception:
            t_db = 293.15

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

        t_set = 273.15 + 37
        t_set = apply_cdu_supply_limit(t_set, y_tcdu_sup, use_control=True)
        model.set("SigTES", sig_tes)
        model.set("TCWDry", t_set)
        model.do_step(current_time, STEP_SIZE, True)

        data["time"].append(current_time / 86400.0)
        data["price"].append(get_electricity_price(current_time, base_date, city_config["region"]))
        data["t_dry_bul"].append((t_db - 273.15) * 9.0 / 5.0 + 32.0)
        data["sig_tes"].append(sig_tes)
        for var in FMU_OUTPUTS:
            data[var].append(model.get(var)[0])

        current_time += STEP_SIZE

    return data


def format_date_ticklabels(day_ticks):
    base_date = datetime(2026, 1, 1)
    return [(base_date + timedelta(days=float(day))).strftime("%b %d") for day in day_ticks]


def draw_figure(city_results):
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

    fig, axes = plt.subplots(3, 2, figsize=(9.2, 5.1), sharex="col")
    fig.subplots_adjust(hspace=0.26, wspace=0.32, left=0.08, right=0.98, bottom=0.12, top=0.96)

    all_temps = []
    all_prices = []
    for _, data in city_results:
        all_temps.extend(data["t_dry_bul"])
        all_prices.extend(data["price"])

    temp_min = np.floor(min(all_temps) - 1.0)
    temp_max = np.ceil(max(all_temps) + 1.0)
    temp_ticks = np.arange(10.0 * np.floor(temp_min / 10.0), 10.0 * np.ceil(temp_max / 10.0) + 0.1, 10.0)
    price_max = max(all_prices) * 1.08
    x_ticks = np.arange(SIM_START_DAY - 1, SIM_END_DAY + 1, 1)
    x_ticklabels = format_date_ticklabels(x_ticks)

    for col, (city_label, data) in enumerate(city_results):
        ax_price, ax_temp, ax_sig = axes[:, col]

        price_line = ax_price.step(
            data["time"],
            data["price"],
            where="post",
            color="#0066ff",
            linewidth=1.6,
            label="TOU price",
        )[0]
        ax_price.set_ylabel("Price")
        ax_price.set_ylim(0.0, price_max * 1.28)

        temp_line = ax_temp.plot(
            data["time"],
            data["t_dry_bul"],
            color="#ff1f1f",
            linewidth=1.5,
            label="Dry-bulb",
        )[0]
        ax_temp.set_ylabel("Temperature")
        ax_temp.set_ylim(temp_min, temp_max + (temp_max - temp_min) * 0.28)
        ax_temp.set_yticks(temp_ticks)

        sig_line = ax_sig.step(
            data["time"],
            data["sig_tes"],
            where="post",
            color="#00b050",
            linewidth=1.6,
            label="SigTES",
        )[0]
        ax_sig.axhline(0.0, color="#999999", linestyle=":", linewidth=0.8)
        ax_sig.set_ylabel("TES behavior")
        ax_sig.set_ylim(-1.05, 1.55)
        ax_sig.set_yticks([-1.0, 0.0, 1.0])
        ax_sig.set_xlabel("Date")

        for ax, handle in ((ax_price, price_line), (ax_temp, temp_line), (ax_sig, sig_line)):
            ax.set_xlim(SIM_START_DAY - 1, SIM_END_DAY)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticklabels)
            ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.35)
            ax.legend(
                [handle],
                [handle.get_label()],
                loc="upper left",
                frameon=True,
                facecolor="white",
                edgecolor="none",
                framealpha=0.92,
            )

    return fig


def main():
    city_results = []
    for city_config in CITY_CONFIGS:
        city_results.append((city_config["label"], run_city_simulation(city_config)))

    fig = draw_figure(city_results)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
