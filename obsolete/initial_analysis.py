import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# ==========================================================
# 1) Scenario / Parameters
# ==========================================================
def generate_scenario(steps=96):
    time_h = np.linspace(0, 24, steps)
    dt_h = 24.0 / steps

    base_load = np.ones(steps) * 70.0
    base_load[36:39] = 135.0
    base_load[48:56] = 130.0
    base_load[68:88] = 130.0

    base_price = 0.5
    peak_price = 1.5
    price = np.ones(steps) * base_price
    price[30:90] = peak_price

    return time_h, dt_h, base_load, price


def default_params():
    return {
        "grid_limit_kw": 100.0,
        "penalty_rate_per_kwh": 10.0,

        "bes_cap_kwh": 100.0,
        "tes_cap_kwh_th": 80.0,

        "tau_up": 1.5,
        "d0": 3.5,
        "k_b": 2.0,
        "k_tes": 0.8,

        "T_safe": 42.0,
        "T_amb": 25.0,
        "phys_params": {"a": 0.04, "b": 0.15, "c": 0.4},
    }


# ==========================================================
# 2) Optimization Core
# ==========================================================
def solve_qp(
    delta_P, u_ref, u_prev, price_k,
    soc, tes_lvl, T_curr, T_safe, T_amb,
    load_k, bes_cap_kwh, tes_cap_kwh, phys_params, dt_h
):
    BES_POWER_LIM = bes_cap_kwh * 0.8
    TES_POWER_LIM = tes_cap_kwh * 0.6

    a, b, c = phys_params["a"], phys_params["b"], phys_params["c"]

    def objective(u):
        p_b, p_t, p_s = u
        J_track = 10.0 * ((p_b - u_ref[0]) ** 2 + (p_t - u_ref[1]) ** 2 + (p_s - u_ref[2]) ** 2)
        J_smooth = 1.0 * ((p_b - u_prev[0]) ** 2 + (p_t - u_prev[1]) ** 2 + (p_s - u_prev[2]) ** 2)
        J_econ = -0.5 * price_k * (p_b + p_t + p_s)
        return J_track + J_smooth + J_econ

    cons = []

    # Peak shaving requirement
    if delta_P > 0:
        cons.append({"type": "ineq", "fun": lambda u: (u[0] + u[1] + u[2]) - delta_P})

    # Battery energy limits
    max_dis_b = max(0.0, (soc - 0.10) * bes_cap_kwh) / dt_h
    max_chg_b = max(0.0, (0.95 - soc) * bes_cap_kwh) / dt_h
    cons.append({"type": "ineq", "fun": lambda u: max_dis_b - u[0]})
    cons.append({"type": "ineq", "fun": lambda u: u[0] + max_chg_b})

    # TES energy limits
    max_dis_t = max(0.0, (tes_lvl - 0.10) * tes_cap_kwh) / dt_h
    max_chg_t = max(0.0, (0.95 - tes_lvl) * tes_cap_kwh) / dt_h
    cons.append({"type": "ineq", "fun": lambda u: max_dis_t - u[1]})
    cons.append({"type": "ineq", "fun": lambda u: u[1] + max_chg_t})

    # Thermal safety
    def thermal_safety_constraint(u):
        p_s = u[2]
        T_next_pred = T_curr + (a * load_k) + (b * p_s) - c * (T_curr - T_amb)
        return (T_safe - 0.10) - T_next_pred

    cons.append({"type": "ineq", "fun": thermal_safety_constraint})

    bnds = (
        (-BES_POWER_LIM, BES_POWER_LIM),
        (-TES_POWER_LIM, TES_POWER_LIM),
        (0.0, 60.0),
    )

    res = minimize(objective, x0=np.array(u_ref, dtype=float),
                   bounds=bnds, constraints=cons, method="SLSQP")

    if not res.success:
        p_b = min(u_ref[0], max_dis_b)
        p_t = min(u_ref[1], max_dis_t)
        p_s = 0.0
        return np.array([p_b, p_t, p_s], dtype=float)

    return res.x


def run_simulation(time_h, dt_h, base_load, price, params):
    steps = len(base_load)

    grid_limit = params["grid_limit_kw"]
    penalty_rate = params["penalty_rate_per_kwh"]
    bes_cap = params["bes_cap_kwh"]
    tes_cap = params["tes_cap_kwh_th"]
    tau_up = params["tau_up"]
    d0 = params["d0"]
    k_b = params["k_b"]
    k_tes = params["k_tes"]
    T_safe = params["T_safe"]
    T_amb = params["T_amb"]
    phys_params = params["phys_params"]

    d_k = 0.0
    soc = 0.6
    tes_lvl = 0.6
    T_curr = 30.0
    u_prev = np.array([0.0, 0.0, 0.0], dtype=float)

    data = {k: [] for k in ["Grid", "BES", "TES", "Setpoint",
                            "d_k", "SOC", "TES_lvl", "T_curr",
                            "Cost_Base", "Cost_Opt", "Penalty"]}

    for k in range(steps):
        load_k = base_load[k]
        price_k = price[k]
        delta_P = load_k - grid_limit

        # duration memory
        if delta_P > 0:
            d_k = min(d_k + dt_h * (1.0 / tau_up), 10.0)
        else:
            d_k = max(d_k - 0.5, 0.0)

        # reference
        u_ref = np.array([0.0, 0.0, 0.0], dtype=float)

        # charging
        if price_k < 0.6 and delta_P <= 0:
            if soc < 0.9:
                u_ref[0] = -0.2 * bes_cap
            if tes_lvl < 0.9:
                u_ref[1] = -0.2 * tes_cap

        # discharging softmax
        if delta_P > 0:
            s_b = 8.0 - k_b * d_k - 2.0 * (1.0 - soc)
            s_t = 6.0 - k_tes * (d_k - d0) ** 2
            s_s = 2.0 + 1.2 * d_k

            scores = np.array([s_b, s_t, s_s], dtype=float)
            exp_scores = np.exp(scores - np.max(scores))
            alphas = exp_scores / np.sum(exp_scores)
            u_ref = alphas * delta_P

        u_opt = solve_qp(
            delta_P, u_ref, u_prev, price_k,
            soc, tes_lvl, T_curr, T_safe, T_amb,
            load_k, bes_cap, tes_cap, phys_params, dt_h
        )
        p_b, p_t, p_s = u_opt

        # state update
        soc = np.clip(soc - (p_b * dt_h) / bes_cap, 0.0, 1.0)
        tes_lvl = np.clip(tes_lvl - (p_t * dt_h) / tes_cap, 0.0, 1.0)

        T_curr = T_curr + (phys_params["a"] * load_k) + (phys_params["b"] * p_s) - phys_params["c"] * (T_curr - T_amb)

        # economics
        p_grid_opt = load_k - np.sum(u_opt)

        cost_base = load_k * price_k * dt_h
        if load_k > grid_limit:
            cost_base += (load_k - grid_limit) * penalty_rate * dt_h

        cost_opt = p_grid_opt * price_k * dt_h
        penalty = 0.0
        if p_grid_opt > grid_limit + 0.01:
            excess = p_grid_opt - grid_limit
            penalty = excess * penalty_rate * dt_h
            cost_opt += penalty

        # record
        data["Grid"].append(p_grid_opt)
        data["BES"].append(p_b)
        data["TES"].append(p_t)
        data["Setpoint"].append(p_s)
        data["d_k"].append(d_k)
        data["SOC"].append(soc)
        data["TES_lvl"].append(tes_lvl)
        data["T_curr"].append(T_curr)
        data["Cost_Base"].append(cost_base)
        data["Cost_Opt"].append(cost_opt)
        data["Penalty"].append(penalty)

        u_prev = u_opt

    for k in data:
        data[k] = np.array(data[k], dtype=float)

    return data


# ==========================================================
# 3) Metrics + Plotting (Matplotlib)
# ==========================================================
def compute_kpis(base_load, res, grid_limit):
    max_base = float(np.max(base_load))
    max_opt = float(np.max(res["Grid"]))
    psr = (max_base - max_opt) / max_base * 100.0

    sum_base = float(np.sum(res["Cost_Base"]))
    sum_opt = float(np.sum(res["Cost_Opt"]))
    ocs = (sum_base - sum_opt) / sum_base * 100.0 if sum_base > 1e-9 else 0.0

    total_penalty = float(np.sum(res["Penalty"]))

    return psr, ocs, total_penalty, max_opt, sum_base, sum_opt


def _stacked_pos_neg_bars(ax, x, series_list, labels, width):
    """
    Stacks bars correctly even if each series contains positive and negative values.
    Draw positives stacked upward; negatives stacked downward.
    """
    pos_base = np.zeros_like(x, dtype=float)
    neg_base = np.zeros_like(x, dtype=float)

    for y, lab in zip(series_list, labels):
        y = np.asarray(y, dtype=float)
        y_pos = np.clip(y, 0, None)
        y_neg = np.clip(y, None, 0)

        ax.bar(x, y_pos, width=width, bottom=pos_base, label=lab)
        ax.bar(x, y_neg, width=width, bottom=neg_base)

        pos_base += y_pos
        neg_base += y_neg


def plot_all_matplotlib(time_h, base_load, price, res, params):
    grid_limit = params["grid_limit_kw"]
    T_safe = params["T_safe"]

    psr, ocs, total_penalty, max_opt, sum_base, sum_opt = compute_kpis(base_load, res, grid_limit)

    # dashboard layout
    fig, axes = plt.subplots(3, 2, figsize=(12, 7))
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]
    ax5, ax6 = axes[2]
    for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
            ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.4)
    # 1) Dispatch + stacked bars
    ax1.plot(time_h, [grid_limit] * len(time_h), linestyle="--", linewidth=2, label="Grid Limit")
    ax1.plot(time_h, base_load, linestyle=":", linewidth=1.5, label="Baseline Load")
    ax1.plot(time_h, res["Grid"], linewidth=2, label="Optimized Grid")

    # bars: use dt-based width in "hours"
    width = (time_h[1] - time_h[0]) * 0.9
    _stacked_pos_neg_bars(
        ax1, time_h,
        [res["BES"], res["TES"], res["Setpoint"]],
        ["BES", "TES", "Setpoint"],
        width=width
    )
    ax1.set_title("Resource Contribution & Grid Compliance")
    ax1.set_xlabel("Time (h)")
    ax1.set_ylabel("Power (kW)")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.1), ncol=6, fontsize=8, frameon=False)

    # 2) Thermal
    ax2.plot(time_h, [T_safe] * len(time_h), linestyle="--", linewidth=2, label="T_safe")
    ax2.plot(time_h, res["T_curr"], linewidth=2, label="Server Temp")
    ax2.fill_between(time_h, 0, res["T_curr"], alpha=0.15)
    ax2.set_title("Temperature Constraint Check")
    ax2.set_xlabel("Time (h)")
    ax2.set_ylabel("Temp (°C)")
    ax2.legend(loc="upper right", fontsize=9)

    # 3) Cumulative cost
    ax3.plot(time_h, np.cumsum(res["Cost_Base"]), linewidth=2, label="Baseline (Cum)")
    ax3.plot(time_h, np.cumsum(res["Cost_Opt"]), linewidth=2, label="Optimized (Cum)")
    ax3.set_title("Cumulative Operational Cost")
    ax3.set_xlabel("Time (h)")
    ax3.set_ylabel("Cost ($)")
    ax3.legend(loc="upper left", fontsize=9)

    # 4) SOC / TES level
    ax4.plot(time_h, res["SOC"], linewidth=2, label="SOC")
    ax4.plot(time_h, res["TES_lvl"], linewidth=2, label="TES Level")
    ax4.set_title("Storage States")
    ax4.set_xlabel("Time (h)")
    ax4.set_ylabel("Normalized Level (0-1)")
    ax4.set_ylim(0, 1)
    ax4.legend(loc="upper right", fontsize=9)

    # 5) Penalty per step
    ax5.bar(time_h, res["Penalty"], width=width, label="Penalty ($/step)")
    ax5.set_title("Penalty Cost (per timestep)")
    ax5.set_xlabel("Time (h)")
    ax5.set_ylabel("$")
    ax5.legend(loc="upper right", fontsize=9)

    # 6) Price curve + KPI text
    ax6.plot(time_h, price, linewidth=2, label="Price ($/kWh)")
    ax6.set_title("Electricity Price + KPI Summary")
    ax6.set_xlabel("Time (h)")
    ax6.set_ylabel("$/kWh")
    ax6.legend(loc="upper right", fontsize=9)

    kpi_text = (
        f"PSR: {psr:.2f}%\n"
        f"OCS: {ocs:.2f}%\n"
        f"Total Penalty: ${total_penalty:.2f}\n"
        f"Grid Peak: {max_opt:.2f} kW (Limit {grid_limit:.0f})\n"
        f"Total Base Cost: ${sum_base:.2f}\n"
        f"Total Opt Cost:  ${sum_opt:.2f}"
    )
    ax6.text(0.02, 0.95, kpi_text, transform=ax6.transAxes, va="top", ha="left",
             bbox=dict(boxstyle="round", alpha=0.15))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ==========================================================
# 4) Main
# ==========================================================
def main():
    params = default_params()
    time_h, dt_h, base_load, price = generate_scenario(steps=96)
    res = run_simulation(time_h, dt_h, base_load, price, params)
    plot_all_matplotlib(time_h, base_load, price, res, params)


if __name__ == "__main__":
    main()
