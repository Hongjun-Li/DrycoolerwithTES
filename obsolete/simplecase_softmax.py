import numpy as np
import matplotlib.pyplot as plt
from pyfmi import load_fmu
from tqdm import tqdm
import cvxpy as cp


def tou_price_2level(hour, low_price=0.5, high_price=2.0, low_window=(0.0, 12.0)):
    h0, h1 = low_window
    hour = hour % 24.0
    return low_price if (h0 <= hour < h1) else high_price


class OnlineSocScaler:
    def __init__(self, looks_like_soc_hi=1.2, eps=1e-9, warmup_steps=30):
        self.looks_like_soc_hi = looks_like_soc_hi
        self.eps = eps
        self.min_seen = np.inf
        self.max_seen = -np.inf
        self.warmup_steps = warmup_steps
        self.n = 0

    def to_norm(self, raw):
        raw = float(raw)
        self.n += 1
        if 0.0 <= raw <= self.looks_like_soc_hi:
            return float(np.clip(raw, 0.0, 1.0))
        self.min_seen = min(self.min_seen, raw)
        self.max_seen = max(self.max_seen, raw)
        span = self.max_seen - self.min_seen
        if span < self.eps or self.n < self.warmup_steps:
            return 0.0
        return float(np.clip((raw - self.min_seen) / span, 0.0, 1.0))




def infer_sign(u_hist, yP_hist_kw):
    u = np.asarray(u_hist, float)
    y = np.asarray(yP_hist_kw, float)
    mask = (np.abs(u) > 0.05)
    if mask.sum() < 30:
        return +1
    u = u[mask]
    y = y[mask]
    dis = (u < -0.05)
    if dis.sum() < 15:
        return +1
    return -1 if float(np.median(y[dis])) < 0 else +1


def run_controller():
    # ---------------- config ----------------
    fmu_name = "ASHRAE26_FMU_Latest_0TES_0FMU.fmu"
    start_time = 0.0
    stop_time = 86400.0 * 1.0
    step_size = 60.0
    dt_hr = step_size / 3600.0

    input_names = {"BES": "sigBES", "TES": "sigTES", "Set": "sigsetpoint"}

    output_keys = [
        "socBES", "socTES",
        "yTserver", "yTCHWret", "yTCDUSup",
        "yPIT", "yPHVAC",
        "yPBES", "yPTES",
    ]

    # ---------------- targets ----------------
    GRID_LIMIT_KW = 500.0
    IT_PEAK_KW = 500.0

    # setpoint
    SETPOINT_NOM_K = 273.15 + 50.0
    T_SAFE_C = 55.0
    T_MARGIN_C = 0.5

    # SOC thresholds / targets
    BES_SOC_MIN, BES_SOC_MAX, BES_SOC_TGT = 0.10, 0.98, 0.90
    TES_SOC_MIN, TES_SOC_MAX, TES_SOC_TGT = 0.20, 0.99, 0.92
    INIT_SOC_TARGET = 0.20

    # resource power caps (kW)
    PBES_DIS_MAX = 120.0
    PBES_CHG_MAX = 120.0
    PTES_DIS_MAX = 120.0
    PTES_CHG_MAX = 120.0

    # input bounds
    U_BES_MAX = 0.8
    U_TES_MAX = 1.0

    # setpoint effect (kW/°C)
    K_SP = 30.0

    # ---------------- price ----------------
    LOW_PRICE = 0.5
    HIGH_PRICE = 2.0
    LOW_WINDOW = (0.0, 12.0)
    PRICE_CHARGE_TH = 0.6
    is_price_low = lambda p: (p <= PRICE_CHARGE_TH)

    # ---------------- setpoint hugging (improved) ----------------
    LIMIT_DB_KW = 2.0

    RAMP_SP_C_BASE = 0.06
    RAMP_SP_C_MAX = 0.25
    RAMP_SP_KW_REF = 80.0

    SP_FILTER_ALPHA = 0.20

    # 更积极释放 setpoint 潜力（仍受 headroom 约束）
    DTSP_PER_HEADROOM = 0.8
    DTSP_POS_CAP_HARD = 10.0
    DTSP_NEG_CAP = 3.0

    # ---------------- TES/BES in u-domain smoothing ----------------
    DU_TES_MAX = 0.06
    DU_BES_MAX = 0.10

    # QP weights - increased to encourage BES/TES participation
    W_TRACK_B = 200.0     # BES: much higher priority
    W_TRACK_T = 150.0     # TES: higher priority
    W_U_SMOOTH_BES = 0.5   # BES: less smooth (faster response)
    W_U_SMOOTH_TES = 2.0   # TES: moderate smoothness
    W_U_MAG_BES = 0.01     # BES: encourage action
    W_U_MAG_TES = 0.05     # TES: encourage action

    # 仍保留 p ramp（有了 u ramp 其实可放宽）
    RAMP_TES_KW = 120.0
    RAMP_BES_KW = 200.0

    # low-price charging encouragement
    W_CHG = 250.0

    # ---------------- online SOC scalers ----------------
    bes_scaler = OnlineSocScaler()
    tes_scaler = OnlineSocScaler()

    # learned gains from history (will be calibrated during first 100 steps)
    G_BES_LEARNED = -100.0  # will be updated online
    G_TES_LEARNED = +100.0  # will be updated online
    u_b_history = []  # track recent u values
    p_b_history = []  # track corresponding p values
    u_t_history = []
    p_t_history = []

    # sign calibration - FIXED (don't auto-infer, too error-prone)
    BES_SIGN = -1  # fixed: negative u -> discharge
    TES_SIGN = -1  # fixed: negative u -> discharge
    sign_calibrated = True  # skip auto inference
    u_b_hist, u_t_hist, yb_hist, yt_hist = [], [], [], []  # still track for debugging

    # internal states
    dTsp_state = 0.0
    dTsp_cmd_prev = 0.0
    d_k = 0.0  # duration memory: tracks how long we've been in overflow

    # ---------------- load FMU ----------------
    model = load_fmu(fmu_name, log_level=3)

    for v in ["socBES", "socTES"]:
        try:
            model.set(v, INIT_SOC_TARGET)
            print(f"[InitSOC] set {v}={INIT_SOC_TARGET}")
        except Exception as e:
            print(f"[InitSOC] cannot set {v} (read-only?), rely on FMU init. msg={e}")

    model.initialize(start_time=start_time, stop_time=stop_time)
    model.set(input_names["Set"], SETPOINT_NOM_K)

    n_steps = int((stop_time - start_time) / step_size)
    res = {k: np.zeros(n_steps) for k in output_keys}
    res["time"] = np.zeros(n_steps)

    log = {
        "Price": np.zeros(n_steps),
        "P_grid_kw": np.zeros(n_steps),
        "P_base_kw": np.zeros(n_steps),
        "BES_u": np.zeros(n_steps),
        "TES_u": np.zeros(n_steps),
        "p_bes_kw": np.zeros(n_steps),
        "p_tes_kw": np.zeros(n_steps),
        "p_sp_kw": np.zeros(n_steps),
        "dTsp": np.zeros(n_steps),
        "SOCBES_n": np.zeros(n_steps),
        "SOCTES_n": np.zeros(n_steps),
        "energy_cost": np.zeros(n_steps),
        "setpoint_cmd_K": np.zeros(n_steps),
        "need_kw": np.zeros(n_steps),
        "dtsp_pos_cap": np.zeros(n_steps),
        "PBES_SCALE": np.zeros(n_steps),
        "PTES_SCALE": np.zeros(n_steps),
        "overflow_kw": np.zeros(n_steps),
        "d_k": np.zeros(n_steps),  # duration memory
        "softmax_bes": np.zeros(n_steps),  # BES softmax weight
        "softmax_tes": np.zeros(n_steps),  # TES softmax weight
        "softmax_sp": np.zeros(n_steps),   # Setpoint softmax weight
        "p_bes_meas": np.zeros(n_steps),   # actual FMU output power (BES)
        "p_tes_meas": np.zeros(n_steps),   # actual FMU output power (TES)
    }

    # prev states
    u_b_prev, u_t_prev = 0.0, 0.0
    p_b_prev, p_t_prev = 0.0, 0.0

    current_time = start_time

    for i in tqdm(range(n_steps)):
        hour = (current_time / 3600.0) % 24.0
        price = tou_price_2level(hour, LOW_PRICE, HIGH_PRICE, LOW_WINDOW)
        lowp = is_price_low(price)

        # last outputs
        if i == 0:
            yPIT_kw = IT_PEAK_KW
            yPHVAC_kw = 30.0
            yPBES_raw = 0.0
            yPTES_raw = 0.0
            socB = 0.0
            socT = 0.0
            Tserver_C = 35.0
        else:
            yPIT_kw = float(res["yPIT"][i - 1]) / 1000.0
            yPHVAC_kw = float(res["yPHVAC"][i - 1]) / 1000.0
            yPBES_raw = float(res["yPBES"][i - 1])
            yPTES_raw = float(res["yPTES"][i - 1])
            socB = float(res["socBES"][i - 1])
            socT = float(res["socTES"][i - 1])
            Tserver_C = float(res["yTserver"][i - 1]) - 273.15

        # scaling (keep your original assumption)
        PBES_SCALE = 1000
        PTES_SCALE = 1000
        yPBES_kw = yPBES_raw / PBES_SCALE
        yPTES_kw = yPTES_raw / PTES_SCALE
        log["PBES_SCALE"][i] = PBES_SCALE
        log["PTES_SCALE"][i] = PTES_SCALE

        socB_n = bes_scaler.to_norm(socB)
        socT_n = tes_scaler.to_norm(socT)

        # sign calibration - DISABLED (use fixed signs)
        u_b_hist.append(u_b_prev)
        u_t_hist.append(u_t_prev)
        yb_hist.append(yPBES_kw)
        yt_hist.append(yPTES_kw)
        # (no auto-inference)

        p_bes_meas_kw = BES_SIGN * yPBES_kw
        p_tes_meas_kw = TES_SIGN * yPTES_kw
        
        # Learn gains from history (update with recent data)
        if len(u_b_history) < 200:
            u_b_history.append(u_b_prev)
            p_b_history.append(p_bes_meas_kw)
            u_t_history.append(u_t_prev)
            p_t_history.append(p_tes_meas_kw)
        else:
            # Keep rolling window
            u_b_history.pop(0)
            p_b_history.pop(0)
            u_t_history.pop(0)
            p_t_history.pop(0)
            u_b_history.append(u_b_prev)
            p_b_history.append(p_bes_meas_kw)
            u_t_history.append(u_t_prev)
            p_t_history.append(p_tes_meas_kw)
        
        # Calculate learned gain: median slope of p/u (excluding near-zero u)
        def estimate_gain(u_list, p_list):
            u_arr = np.array(u_list)
            p_arr = np.array(p_list)
            mask = np.abs(u_arr) > 0.05  # only use significant movements
            if mask.sum() < 10:
                return None
            ratios = p_arr[mask] / (u_arr[mask] + 1e-9)
            # Filter outliers
            med = np.median(ratios)
            mad = np.median(np.abs(ratios - med)) + 1e-9
            ratios_clean = ratios[np.abs(ratios - med) < 3.0 * mad]
            if len(ratios_clean) > 0:
                return float(np.median(ratios_clean))
            return None
        
        g_b_new = estimate_gain(u_b_history, p_b_history)
        g_t_new = estimate_gain(u_t_history, p_t_history)
        
        if g_b_new is not None and abs(g_b_new) > 10:  # sanity check
            G_BES_LEARNED = 0.7 * G_BES_LEARNED + 0.3 * g_b_new  # exponential smoothing
        if g_t_new is not None and abs(g_t_new) > 10:
            G_TES_LEARNED = 0.7 * G_TES_LEARNED + 0.3 * g_t_new
        
        G_b = G_BES_LEARNED
        G_t = G_TES_LEARNED

        # base power
        P_base_kw = yPIT_kw + yPHVAC_kw
        log["P_base_kw"][i] = P_base_kw

        # dynamic thermal cap for +dTsp
        headroom_C = (T_SAFE_C - T_MARGIN_C) - Tserver_C
        dTsp_pos_cap = float(np.clip(headroom_C * DTSP_PER_HEADROOM, 0.0, DTSP_POS_CAP_HARD))
        log["dtsp_pos_cap"][i] = dTsp_pos_cap

        # SOC gated power limits
        pb_dis_max = PBES_DIS_MAX if socB_n > BES_SOC_MIN + 1e-3 else 0.0
        pt_dis_max = PTES_DIS_MAX if socT_n > TES_SOC_MIN + 1e-3 else 0.0
        pb_chg_max = PBES_CHG_MAX if socB_n < BES_SOC_MAX - 1e-3 else 0.0
        pt_chg_max = PTES_CHG_MAX if socT_n < TES_SOC_MAX - 1e-3 else 0.0

        # ---------- 1) Setpoint hugging: always track grid limit ----------
        p_sp_now = K_SP * dTsp_state
        P_grid_no_storage = P_base_kw - p_sp_now

        err_kw = P_grid_no_storage - GRID_LIMIT_KW
        if abs(err_kw) <= LIMIT_DB_KW:
            dTsp_des = dTsp_cmd_prev
        else:
            need_kw = max(0.0, err_kw)
            dTsp_des = need_kw / max(K_SP, 1e-6)

        dTsp_des = float(np.clip(dTsp_des, 0.0, dTsp_pos_cap))

        # adaptive ramp (more overflow -> faster ramp)
        need_kw_now = max(0.0, P_grid_no_storage - GRID_LIMIT_KW)
        ramp_sp = RAMP_SP_C_BASE + (RAMP_SP_C_MAX - RAMP_SP_C_BASE) * float(
            np.clip(need_kw_now / max(RAMP_SP_KW_REF, 1e-6), 0.0, 1.0)
        )

        dTsp_des = float(np.clip(dTsp_des, dTsp_cmd_prev - ramp_sp, dTsp_cmd_prev + ramp_sp))

        dTsp_state = (1.0 - SP_FILTER_ALPHA) * dTsp_state + SP_FILTER_ALPHA * dTsp_des
        dTsp_cmd = float(np.clip(dTsp_state, -DTSP_NEG_CAP, dTsp_pos_cap))
        dTsp_cmd_prev = dTsp_cmd

        p_sp_cmd = K_SP * dTsp_cmd

        # remaining overflow after setpoint
        P_grid_after_sp = P_base_kw - p_sp_cmd
        overflow_kw = max(0.0, P_grid_after_sp - GRID_LIMIT_KW)

        # ---------- UPDATE DURATION MEMORY ----------
        # Track how long we've been in overflow (0-1 day scale)
        if overflow_kw > 0.1:
            d_k = min(d_k + dt_hr * (1.0 / 1.5), 10.0)  # 1.5hr ramp-up time
        else:
            d_k = max(d_k - 0.5 * dt_hr, 0.0)  # faster decay
        
        log["d_k"][i] = d_k
        log["need_kw"][i] = overflow_kw
        log["overflow_kw"][i] = overflow_kw

        # ---------- 2) SOFTMAX-BASED HIERARCHICAL ALLOCATION ----------
        # Compute scores for each resource based on duration memory (from initial_analysis.py strategy)
        k_b_coef = 1.5    # BES sensitivity to duration
        k_tes_coef = 1.2  # TES sensitivity (parabolic around d0)
        d0 = 3.0          # optimal duration for TES (hours)
        
        if overflow_kw > 0.1:
            # Discharging: BES fast-respond, TES gradual, Setpoint slowest
            score_b = 8.0 - k_b_coef * d_k - 2.0 * (1.0 - socB_n)  # BES: high initially, decays
            score_t = 6.0 - k_tes_coef * (d_k - d0) ** 2            # TES: peaks at d_k=d0
            score_s = 2.0 + 1.2 * d_k                              # Setpoint: low initially, grows
            
            scores = np.array([max(score_b, 0.01), max(score_t, 0.01), max(score_s, 0.01)])
            exp_scores = np.exp(scores - np.max(scores))
            alphas = exp_scores / np.sum(exp_scores)
            
            # BES handles first slice
            p_b_shave_tgt = float(alphas[0] * min(overflow_kw, pb_dis_max))
            # TES handles second slice
            rem_after_b = max(0.0, overflow_kw - p_b_shave_tgt)
            p_t_shave_tgt = float(alphas[1] * min(rem_after_b, pt_dis_max))
        else:
            # No overflow: zero discharge targets
            alphas = np.array([0.33, 0.33, 0.34])
            p_b_shave_tgt = 0.0
            p_t_shave_tgt = 0.0
        
        log["softmax_bes"][i] = alphas[0]
        log["softmax_tes"][i] = alphas[1]
        log["softmax_sp"][i] = alphas[2]

        # low price charging: only if not causing overflow
        p_b_chg_tgt = 0.0
        p_t_chg_tgt = 0.0
        if lowp:
            P_after_dis = P_base_kw - p_sp_cmd - p_t_shave_tgt - p_b_shave_tgt
            headroom_kw = max(0.0, GRID_LIMIT_KW - P_after_dis)

            want_b = float(np.clip((BES_SOC_TGT - socB_n) / 0.25, 0.0, 1.0))
            want_t = float(np.clip((TES_SOC_TGT - socT_n) / 0.25, 0.0, 1.0))
            s = want_b + want_t + 1e-9

            p_b_chg_tgt = -min(pb_chg_max * want_b, headroom_kw * (want_b / s))
            p_t_chg_tgt = -min(pt_chg_max * want_t, headroom_kw * (want_t / s))

        p_b_tgt = p_b_shave_tgt + p_b_chg_tgt
        p_t_tgt = p_t_shave_tgt + p_t_chg_tgt

        # ---------- 3) QP in u-domain: smooth sigTES (and sigBES) ----------
        u_b = cp.Variable()
        u_t = cp.Variable()

        p_b_pred = G_b * u_b
        p_t_pred = G_t * u_t

        obj = 0
        obj += W_TRACK_B * cp.square(p_b_pred - p_b_tgt) + W_TRACK_T * cp.square(p_t_pred - p_t_tgt)
        obj += W_U_SMOOTH_BES * cp.square(u_b - u_b_prev) + W_U_SMOOTH_TES * cp.square(u_t - u_t_prev)
        obj += W_U_MAG_BES * cp.square(u_b) + W_U_MAG_TES * cp.square(u_t)

        cons = []
        cons += [u_b <= U_BES_MAX, u_b >= -U_BES_MAX]
        cons += [u_t <= U_TES_MAX, u_t >= -U_TES_MAX]
        cons += [u_b - u_b_prev <= DU_BES_MAX, u_b_prev - u_b <= DU_BES_MAX]
        cons += [u_t - u_t_prev <= DU_TES_MAX, u_t_prev - u_t <= DU_TES_MAX]
        cons += [p_b_pred <= pb_dis_max, p_b_pred >= -pb_chg_max]
        cons += [p_t_pred <= pt_dis_max, p_t_pred >= -pt_chg_max]
        cons += [p_b_pred - p_b_prev <= RAMP_BES_KW, p_b_prev - p_b_pred <= RAMP_BES_KW]
        cons += [p_t_pred - p_t_prev <= RAMP_TES_KW, p_t_prev - p_t_pred <= RAMP_TES_KW]

        prob = cp.Problem(cp.Minimize(obj), cons)
        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            if prob.status not in ("optimal", "optimal_inaccurate"):
                raise RuntimeError(prob.status)
            u_b_cmd = float(u_b.value)
            u_t_cmd = float(u_t.value)
        except Exception:
            u_b_cmd = 0.0 if abs(G_b) < 1e-6 else float(p_b_tgt / G_b)
            u_t_cmd = 0.0 if abs(G_t) < 1e-6 else float(p_t_tgt / G_t)
            # enforce slew in fallback
            u_b_cmd = float(np.clip(u_b_cmd, u_b_prev - DU_BES_MAX, u_b_prev + DU_BES_MAX))
            u_t_cmd = float(np.clip(u_t_cmd, u_t_prev - DU_TES_MAX, u_t_prev + DU_TES_MAX))

        u_b_cmd = float(np.clip(u_b_cmd, -U_BES_MAX, U_BES_MAX))
        u_t_cmd = float(np.clip(u_t_cmd, -U_TES_MAX, U_TES_MAX))

        # back-calc commanded power for logs
        p_b_cmd = float(G_b * u_b_cmd)
        p_t_cmd = float(G_t * u_t_cmd)

        # write inputs
        set_cmd = SETPOINT_NOM_K + dTsp_cmd
        model.set(input_names["BES"], u_b_cmd)
        model.set(input_names["TES"], u_t_cmd)
        model.set(input_names["Set"], set_cmd)
        model.do_step(current_t=current_time, step_size=step_size)

        vals = model.get(output_keys)
        for idx, k in enumerate(output_keys):
            res[k][i] = vals[idx]
        res["time"][i] = current_time

        # logs
        log["Price"][i] = price
        log["SOCBES_n"][i] = socB_n
        log["SOCTES_n"][i] = socT_n
        log["BES_u"][i] = u_b_cmd
        log["TES_u"][i] = u_t_cmd
        log["p_bes_kw"][i] = p_b_cmd
        log["p_tes_kw"][i] = p_t_cmd
        log["p_bes_meas"][i] = p_bes_meas_kw  # actual FMU output
        log["p_tes_meas"][i] = p_tes_meas_kw  # actual FMU output
        log["dTsp"][i] = dTsp_cmd
        log["p_sp_kw"][i] = p_sp_cmd
        log["setpoint_cmd_K"][i] = set_cmd

        # Use actual FMU outputs for grid power calculation
        P_grid_proxy = float(P_base_kw - p_bes_meas_kw - p_tes_meas_kw - p_sp_cmd)
        log["P_grid_kw"][i] = P_grid_proxy
        log["energy_cost"][i] = price * P_grid_proxy * dt_hr

        # update prev
        u_b_prev, u_t_prev = u_b_cmd, u_t_cmd
        p_b_prev, p_t_prev = p_b_cmd, p_t_cmd
        current_time += step_size

    return res, log, GRID_LIMIT_KW, step_size, SETPOINT_NOM_K


def plot_results(res, log, grid_limit_kw, step_size):
    t_hr = res["time"] / 3600.0
    dt_hr = step_size / 3600.0

    P_it = res["yPIT"] / 1000.0
    P_hvac = res["yPHVAC"] / 1000.0
    P_base = P_it + P_hvac
    P_opt = P_base - log["p_bes_meas"] - log["p_tes_meas"] - log["p_sp_kw"]

    base_peak = float(np.max(P_base))
    opt_peak = float(np.max(P_opt))
    psr = (base_peak - opt_peak) / max(base_peak, 1e-6) * 100.0
    base_cost = float(np.sum(log["Price"] * P_base * dt_hr))
    opt_cost = float(np.sum(log["energy_cost"]))

    w = dt_hr * 0.9
    fig, axs = plt.subplots(3, 2, figsize=(10, 7), sharex=True)
    axs = axs.flatten()

    axs[0].plot(t_hr, np.ones_like(t_hr) * grid_limit_kw, linestyle="--", label="Grid Limit")
    axs[0].plot(t_hr, P_base, linestyle=":", label="Baseline Load")
    axs[0].plot(t_hr, P_opt, linewidth=2, label="Optimized Grid")
    axs[0].bar(t_hr, log["p_bes_meas"], width=w, label="BES")
    axs[0].bar(t_hr, log["p_tes_meas"], width=w, label="TES")
    axs[0].bar(t_hr, log["p_sp_kw"], width=w, label="Setpoint")
    axs[0].set_title("Resource Contribution & Grid Compliance")
    axs[0].set_ylabel("Power (kW)")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="center", ncol=3, fontsize="small")

    axs[1].plot(t_hr, np.ones_like(t_hr) * 55.0, linestyle="--", label="T_safe (ref)")
    axs[1].plot(t_hr, res["yTserver"] - 273.15, label="Server Temp")
    axs[1].plot(t_hr, res["yTCHWret"] - 273.15, label="Tchwret")
    axs[1].plot(t_hr, res["yTCDUSup"] - 273.15, label="Tcdusup")
    axs[1].set_title("Temperature Constraint Check")
    axs[1].set_ylabel("Temp (°C)")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="best", fontsize="small")

    axs[2].plot(t_hr, log["SOCBES_n"], label="SOC (BES)")
    axs[2].plot(t_hr, log["SOCTES_n"], label="SOC (TES)")
    axs[2].set_title("Storage States")
    axs[2].set_ylabel("Normalized Level (0-1)")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend()

    axs[3].plot(t_hr, np.cumsum(log["Price"] * P_base * dt_hr), label="Baseline (Cum)")
    axs[3].plot(t_hr, np.cumsum(log["energy_cost"]), label="Optimized (Cum)")
    axs[3].set_title("Cumulative Operational Cost")
    axs[3].set_ylabel("Cost (proxy)")
    axs[3].grid(True, alpha=0.3)
    axs[3].legend()

    ax = axs[4]
    ax.plot(t_hr, log["BES_u"], label="sigBES (u)")
    ax.plot(t_hr, log["TES_u"], label="sigTES (u)")
    ax.set_title("FMU Input Signals")
    ax.set_ylabel("u")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(t_hr, log["setpoint_cmd_K"], linestyle="--", label="sigsetpoint (K)")
    ax2.set_ylabel("K")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", fontsize="small")

    axs[5].plot(t_hr, log["Price"], label="Price (rel)")
    txt = (
        f"PSR: {psr:.2f}%\n"
        f"Grid Peak: {opt_peak:.2f} kW (Limit {grid_limit_kw:.0f})\n"
        f"Total Base Cost: {base_cost:.2f}\n"
        f"Total Opt Cost: {opt_cost:.2f}\n"
        f"PBES scale(raw->kW): ~{np.median(log['PBES_SCALE']):.0f}\n"
        f"PTES scale(raw->kW): ~{np.median(log['PTES_SCALE']):.0f}"
    )
    axs[5].text(0.02, 0.98, txt, transform=axs[5].transAxes, va="top",
                bbox=dict(boxstyle="round", alpha=0.15))
    axs[5].set_title("Electricity Price + KPI Summary")
    axs[5].set_ylabel("Price (rel)")
    axs[5].grid(True, alpha=0.3)
    axs[5].legend(loc="upper right")

    for axx in axs:
        axx.set_xlabel("Time (h)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    res, log, grid_limit_kw, step_size, setpoint_nom_k = run_controller()
    plot_results(res, log, grid_limit_kw, step_size)
