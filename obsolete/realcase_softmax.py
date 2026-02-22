#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from pyfmi import load_fmu
from tqdm import tqdm


class SoftmaxCoordinationController:
    """
    Softmax-based Multi-Resource Coordination Control
    Inspired by initial_analysis.py, adapted for FMU real-time control
    
    Resources: BES (Battery), TES (Thermal), Setpoint (HVAC adjustment)
    Strategy:
      - Overload Duration Tracking: d(t) accumulates when overflow > 0
      - Softmax Weights: Allocation based on d(t), SOC availability
      - Charging: Only allowed during low-price periods (off-peak 23-12)
      - Discharging: Triggered by overflow, allocated via Softmax weights
    """
    
    def __init__(self, grid_limit_kw=500.0, dt_sec=300.0):
        # ===== Overload Duration Parameters =====
        self.tau_up = 1.5  # Rise time constant (accumulation)
        self.d_prev = 0.0  # Previous overload duration
        
        # ===== Softmax Parameters (from initial_analysis.py) =====
        self.k_b = 2.0    # BES sensitivity to d(t)
        self.k_tes = 0.8  # TES sensitivity to (d-d0)^2
        self.d0 = 2.0     # Optimal d for TES
        
        # ===== Resource Capacities & Limits =====
        self.grid_limit_kw = grid_limit_kw
        self.dt_hr = dt_sec / 3600.0
        self.bes_cap_kw = 50.0  # Max 50 kW discharge/charge rate
        self.tes_cap_kw = 50.0
        
        # ===== Setpoint Parameters =====
        self.setpoint_base_c = 50.0
        self.setpoint_max_c = 60.0
        self.k_p_temp = 0.8  # Power reduction per °C setpoint raise
        
        # ===== State Memory =====
        self.u_bes_prev = 0.0
        self.u_tes_prev = 0.0
        self.setpoint_prev_c = self.setpoint_base_c
    
    def estimate_overload_duration(self, delta_P, dt_hr):
        """
        Track overload persistence d(t) using first-order dynamics
        d_k = d_prev + dt/tau_up when overflow > 0
        d_k = max(d_prev - decay, 0) when overflow <= 0
        """
        if delta_P > 0:
            # Accumulation: slower rise time (tau_up = 1.5 hours)
            d_new = self.d_prev + dt_hr * (1.0 / self.tau_up)
            d_new = min(d_new, 10.0)  # Cap at 10
        else:
            # Decay: faster dissipation
            d_new = max(self.d_prev - 0.5, 0.0)
        
        self.d_prev = d_new
        return d_new
    
    def softmax_weights_discharge(self, d_overload, socBES, socTES):
        """
        Allocate overflow reduction across BES, TES, and Setpoint
        using Softmax based on overload duration and SOC availability
        
        Returns: (w_bes, w_tes, w_sp) normalized weights summing to 1
        """
        # BES: Fast response, penalized if SOC is low
        s_bes = 6.0 - self.k_b * d_overload - 3.0 * (1.0 - socBES)
        
        # TES: Slower response, peaks at d = d0
        s_tes = 5.0 - self.k_tes * (d_overload - self.d0) ** 2 - 2.0 * (1.0 - socTES)
        
        # Setpoint: More aggressive - always available, no SOC limit
        s_sp = 5.0 + 2.0 * d_overload
        
        # Softmax with numerical stability
        scores = np.array([s_bes, s_tes, s_sp], dtype=float)
        scores_max = np.max(scores)
        exp_scores = np.exp(scores - scores_max)
        weights = exp_scores / np.sum(exp_scores)
        
        return weights[0], weights[1], weights[2]
    
    def control_step(self, P_IT_kw, P_HVAC_kw, socBES, socTES, 
                    current_time_sec=0.0):
        """
        Execute one control step using Softmax+Smooth coordination
        """
        # ===== SOC Constraints =====
        BES_SOC_MIN, BES_SOC_MAX = 0.1, 0.9
        TES_SOC_MIN, TES_SOC_MAX = 0.2, 0.9
        
        # ===== Time-based Pricing =====
        hour = (current_time_sec / 3600.0) % 24.0
        is_off_peak = 23 <= hour or hour < 12  # Low price: 23-12
        
        # ===== Calculate Overflow =====
        P_base = P_IT_kw + P_HVAC_kw
        delta_P = P_base - self.grid_limit_kw
        overflow = max(0.0, delta_P)
        
        # ===== Overload Duration Estimation =====
        d_overload = self.estimate_overload_duration(overflow, self.dt_hr)
        
        # ===== Generate Reference Commands =====
        u_bes_ref = 0.0
        u_tes_ref = 0.0
        setpoint_ref_c = self.setpoint_base_c
        
        # **DISCHARGE MODE**: Allocate overflow using Softmax weights
        if overflow > 0.0:
            # Compute Softmax weights based on d(t), SOC
            w_bes, w_tes, w_sp = self.softmax_weights_discharge(
                d_overload, socBES, socTES
            )
            
            # Available discharge capacity - strict SOC protection
            # Only allow discharge if SOC > MIN + safety margin (0.05)
            max_bes_discharge = max(0.0, (socBES - BES_SOC_MIN - 0.05) * self.bes_cap_kw) if socBES > BES_SOC_MIN + 0.05 else 0.0
            max_tes_discharge = max(0.0, (socTES - TES_SOC_MIN - 0.05) * self.tes_cap_kw) if socTES > TES_SOC_MIN + 0.05 else 0.0
            max_setpoint_saving = self.k_p_temp * (self.setpoint_max_c - self.setpoint_base_c)
            
            # Allocate to each resource (proportional but capped)
            p_bes_alloc = min(w_bes * overflow, max_bes_discharge)
            p_tes_alloc = min(w_tes * overflow, max_tes_discharge)
            p_sp_alloc = min(w_sp * overflow, max_setpoint_saving)
            
            # Redistribute unmet overflow
            unmet = overflow - (p_bes_alloc + p_tes_alloc + p_sp_alloc)
            if unmet > 0.1:
                extra_bes = min(unmet * 0.7, max_bes_discharge - p_bes_alloc)
                p_bes_alloc += extra_bes
                unmet -= extra_bes
                
                extra_tes = min(unmet, max_tes_discharge - p_tes_alloc)
                p_tes_alloc += extra_tes
            
            # Convert power to control signals
            u_bes_ref = -p_bes_alloc / self.bes_cap_kw  # Negative = discharge
            u_tes_ref = -p_tes_alloc / self.tes_cap_kw  # Negative = discharge
            
            # Setpoint adjustment
            if p_sp_alloc > 0.1:
                setpoint_ref_c = self.setpoint_base_c + (p_sp_alloc / self.k_p_temp)
        
        # **CHARGING MODE**: Only during low-price periods (off-peak)
        else:
            if is_off_peak and P_base < self.grid_limit_kw:
                # Charge BES and TES in parallel (not sequentially)
                available_headroom = self.grid_limit_kw - P_base
                
                # ===== BES Charging =====
                if socBES < BES_SOC_MAX - 0.05:
                    bes_charge = min(0.3, available_headroom / (2.0 * self.bes_cap_kw))
                    u_bes_ref = bes_charge  # Positive = charge
                
                # ===== TES Charging (parallel, not conditional on BES) =====
                if socTES < TES_SOC_MAX - 0.05:
                    tes_charge = min(0.3, available_headroom / (2.0 * self.tes_cap_kw))
                    u_tes_ref = tes_charge  # Positive = charge
            
            setpoint_ref_c = self.setpoint_base_c
        
        # ===== Smooth Transitions (stronger smoothing to reduce spikes) =====
        # Off-peak: 65% history, Peak: 70% history (more conservative)
        alpha_bes = 0.65 if is_off_peak else 0.7
        alpha_tes = 0.65 if is_off_peak else 0.7
        
        u_bes_opt = alpha_bes * self.u_bes_prev + (1.0 - alpha_bes) * u_bes_ref
        u_tes_opt = alpha_tes * self.u_tes_prev + (1.0 - alpha_tes) * u_tes_ref
        
        # ===== Final SOC-Based Clipping - Strict Protection =====
        # Prevent any discharge when SOC at or below minimum + safety margin
        if socBES <= BES_SOC_MIN + 0.03:
            u_bes_opt = max(0.0, u_bes_opt)  # No discharge
        elif socBES >= BES_SOC_MAX - 0.03:
            u_bes_opt = min(0.0, u_bes_opt)  # No charge
        
        if socTES <= TES_SOC_MIN + 0.03:
            u_tes_opt = max(0.0, u_tes_opt)  # No discharge
        elif socTES >= TES_SOC_MAX - 0.03:
            u_tes_opt = min(0.0, u_tes_opt)  # No charge
        
        # Clamp to [-1, 1]
        u_bes_opt = float(np.clip(u_bes_opt, -1.0, 1.0))
        u_tes_opt = float(np.clip(u_tes_opt, -1.0, 1.0))
        
        # ===== Setpoint Smoothing (stronger damping to reduce HVAC spikes) =====
        setpoint_target_k = 273.15 + setpoint_ref_c
        setpoint_cmd = 0.9 * (273.15 + self.setpoint_prev_c) + 0.1 * setpoint_target_k
        
        # ===== Update State Memory =====
        self.u_bes_prev = u_bes_opt
        self.u_tes_prev = u_tes_opt
        self.setpoint_prev_c = setpoint_cmd - 273.15
        
        return u_bes_opt, u_tes_opt, setpoint_cmd, d_overload, overflow


def run_fmu_simulation():
    """FMU Co-simulation using Softmax Coordination Controller"""
    
    # ============ Configuration ============
    fmu_name = "ASHRAE26_FMU_Latest_0TES_0FMU.fmu"
    start_time = 0.0
    stop_time = 86400.0 * 2.0  # 2 days
    step_size = 300.0  # 5 minutes
    
    input_names = {
        "BES": "sigBES",
        "TES": "sigTES",
        "Set": "sigsetpoint"
    }
    
    output_names = [
        "socBES", "socTES",
        "yTserver", "yTCHWret", "yTCDUSup",
        "yPIT", "yPHVAC",
        "yPBES", "yPTES",
    ]
    
    GRID_LIMIT_KW = 500.0
    IT_PEAK_KW = 500.0
    
    # ============ Load and Initialize FMU ============
    print(f"Loading FMU: {fmu_name}")
    model = load_fmu(fmu_name, log_level=3)
    
    model.initialize(start_time=start_time, stop_time=stop_time)
    model.set(input_names["Set"], 273.15 + 50.0)
    
    n_steps = int((stop_time - start_time) / step_size)
    print(f"Running {n_steps} steps ({n_steps*step_size/86400:.1f} days)...")
    
    # Initialize arrays
    outputs = {name: np.zeros(n_steps) for name in output_names}
    outputs["time"] = np.zeros(n_steps)
    
    log = {
        "P_IT_kw": np.zeros(n_steps),
        "P_HVAC_kw": np.zeros(n_steps),
        "P_HVAC_baseline_kw": np.zeros(n_steps),
        "P_BES_kw": np.zeros(n_steps),
        "P_TES_kw": np.zeros(n_steps),
        "u_BES": np.zeros(n_steps),
        "u_TES": np.zeros(n_steps),
        "T_server_C": np.zeros(n_steps),
        "T_CHWret_C": np.zeros(n_steps),
        "T_CDUsup_C": np.zeros(n_steps),
        "P_grid_kw": np.zeros(n_steps),
        "setpoint_cmd_C": np.zeros(n_steps),
        "overflow_kw": np.zeros(n_steps),
        "d_overload": np.zeros(n_steps),
        "P_setpoint_only": np.zeros(n_steps),
        "P_bes_setpoint": np.zeros(n_steps),
        "P_bes_tes_setpoint": np.zeros(n_steps),
    }
    
    # ============ Initialize Controller =====
    controller = SoftmaxCoordinationController(grid_limit_kw=GRID_LIMIT_KW, dt_sec=step_size)
    
    setpoint_log = []
    u_bes_log = []
    u_tes_log = []
    
    current_time = start_time
    
    # ============ Main Control Loop ============
    for i in tqdm(range(n_steps), desc="Main simulation"):
        # Read previous state
        if i == 0:
            yPIT_kw = IT_PEAK_KW * 0.95
            yPHVAC_kw = 30.0
            socBES = 0.5
            socTES = 0.5
        else:
            yPIT_kw = float(outputs["yPIT"][i-1]) / 1000.0
            yPHVAC_kw = float(outputs["yPHVAC"][i-1]) / 1000.0
            socBES = float(outputs["socBES"][i-1])
            socTES = float(outputs["socTES"][i-1])
        
        # Control step
        u_bes_cmd, u_tes_cmd, setpoint_cmd_K, d_overload, overflow = controller.control_step(
            P_IT_kw=yPIT_kw,
            P_HVAC_kw=yPHVAC_kw,
            socBES=socBES,
            socTES=socTES,
            current_time_sec=current_time
        )
        
        # Set FMU inputs
        model.set(input_names["BES"], u_bes_cmd)
        model.set(input_names["TES"], u_tes_cmd)
        model.set(input_names["Set"], setpoint_cmd_K)
        
        # Execute step
        model.do_step(current_t=current_time, step_size=step_size)
        
        # Read outputs
        vals = model.get(output_names)
        
        for idx, name in enumerate(output_names):
            outputs[name][i] = vals[idx]
        outputs["time"][i] = current_time
        
        # Log data
        log["P_IT_kw"][i] = yPIT_kw
        log["P_HVAC_kw"][i] = yPHVAC_kw
        log["P_BES_kw"][i] = float(outputs["yPBES"][i]) / 1000.0 if abs(float(outputs["yPBES"][i])) > 100 else float(outputs["yPBES"][i])
        log["P_TES_kw"][i] = float(outputs["yPTES"][i]) / 1000.0 if abs(float(outputs["yPTES"][i])) > 100 else float(outputs["yPTES"][i])
        log["u_BES"][i] = u_bes_cmd
        log["u_TES"][i] = u_tes_cmd
        log["T_server_C"][i] = float(outputs["yTserver"][i]) - 273.15
        log["T_CHWret_C"][i] = float(outputs["yTCHWret"][i]) - 273.15
        log["T_CDUsup_C"][i] = float(outputs["yTCDUSup"][i]) - 273.15
        
        P_grid_kw = yPIT_kw + yPHVAC_kw + log["P_BES_kw"][i] + log["P_TES_kw"][i]
        log["P_grid_kw"][i] = P_grid_kw
        log["setpoint_cmd_C"][i] = setpoint_cmd_K - 273.15
        log["overflow_kw"][i] = overflow
        log["d_overload"][i] = d_overload
        
        # Progressive effects
        log["P_setpoint_only"][i] = yPIT_kw + yPHVAC_kw
        log["P_bes_setpoint"][i] = yPIT_kw + yPHVAC_kw + log["P_BES_kw"][i]
        log["P_bes_tes_setpoint"][i] = P_grid_kw
        
        # Store for baseline
        current_time += step_size
        setpoint_log.append(setpoint_cmd_K)
        u_bes_log.append(u_bes_cmd)
        u_tes_log.append(u_tes_cmd)
    
    # ============ Baseline Simulation ============
    print("\nComputing baseline (fixed 50°C)...")
    model_baseline = load_fmu(fmu_name, log_level=3)
    model_baseline.initialize(start_time=start_time, stop_time=stop_time)
    model_baseline.set(input_names["Set"], 273.15 + 50.0)
    
    current_time = start_time
    for i in tqdm(range(n_steps), desc="Baseline simulation"):
        model_baseline.set(input_names["BES"], u_bes_log[i])
        model_baseline.set(input_names["TES"], u_tes_log[i])
        model_baseline.set(input_names["Set"], 273.15 + 50.0)
        
        model_baseline.do_step(current_t=current_time, step_size=step_size)
        
        vals_baseline = model_baseline.get(output_names)
        yPHVAC_baseline_kw = float(vals_baseline[output_names.index("yPHVAC")]) / 1000.0
        log["P_HVAC_baseline_kw"][i] = yPHVAC_baseline_kw
        
        current_time += step_size
    
    return outputs, log, GRID_LIMIT_KW, step_size


def plot_results(outputs, log, grid_limit_kw, step_size):
    """Plot 8-panel dashboard"""
    t_hr = outputs["time"] / 3600.0
    dt_hr = step_size / 3600.0
    
    P_baseline = log["P_IT_kw"] + log["P_HVAC_kw"]
    P_optimized = log["P_bes_tes_setpoint"]
    
    peak_baseline = np.max(P_baseline)
    peak_optimized = np.max(P_optimized)
    psr = (peak_baseline - peak_optimized) / peak_baseline * 100.0
    
    fig, axs = plt.subplots(4, 2, figsize=(16, 8))
    axs = axs.flatten()
    
    # [0] Grid Power
    axs[0].plot(t_hr, P_baseline, label="Baseline", linewidth=1, color='gray', linestyle='--')
    axs[0].plot(t_hr, log["P_bes_tes_setpoint"], label="Optimized", linewidth=1, color='red')
    axs[0].axhline(grid_limit_kw, color='black', linestyle=':', linewidth=1, label=f"Limit ({grid_limit_kw:.0f} kW)")
    axs[0].set_title("Grid Power (Softmax Coordination)")
    axs[0].set_ylabel("Power (kW)")
    axs[0].legend(fontsize=8)
    axs[0].grid(True, alpha=0.3)
    
    # [1] Storage Power
    axs[1].plot(t_hr, log["P_BES_kw"], label="BES", linewidth=1.5)
    axs[1].plot(t_hr, log["P_TES_kw"], label="TES", linewidth=1.5)
    axs[1].axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    axs[1].set_title("Storage Power (Softmax Allocated)")
    axs[1].set_ylabel("Power (kW)")
    axs[1].legend(fontsize=8)
    axs[1].grid(True, alpha=0.3)
    
    # [2] SOC
    axs[2].plot(t_hr, outputs["socBES"], label="BES SOC", linewidth=1.5)
    axs[2].plot(t_hr, outputs["socTES"], label="TES SOC", linewidth=1.5)
    axs[2].set_title("Storage SOC (w/ Limits)")
    axs[2].set_ylabel("SOC (0-1)")
    axs[2].set_ylim(0, 1)
    axs[2].legend(fontsize=8)
    axs[2].grid(True, alpha=0.3)
    
    # [3] Power Contributions (relative to baseline)
    P_baseline_full = log["P_IT_kw"] + log["P_HVAC_baseline_kw"]
    P_setpoint_contrib = np.maximum(P_baseline_full - log["P_setpoint_only"], 0)
    P_bes_contrib = np.maximum(log["P_setpoint_only"] - log["P_bes_setpoint"], 0)
    P_tes_contrib = np.maximum(log["P_bes_setpoint"] - log["P_bes_tes_setpoint"], 0)
    
    axs[3].stackplot(t_hr, P_setpoint_contrib, P_bes_contrib, P_tes_contrib,
                     labels=["Setpoint", "BES", "TES"],
                     colors=['#ff9999', '#66b3ff', '#99ff99'], alpha=0.7)
    axs[3].set_title("Power Reduction Contributions (vs Baseline)")
    axs[3].set_ylabel("Power Reduction (kW)")
    axs[3].legend(fontsize=8, loc='upper left')
    axs[3].grid(True, alpha=0.3)
    
    # [4] Temperature
    axs[4].plot(t_hr, log["T_server_C"], label="Server", linewidth=1.5)
    axs[4].axhline(60, color='r', linestyle='--', linewidth=1, label="T_safe (60°C)")
    axs[4].set_title("Temperature Profile")
    axs[4].set_ylabel("Temperature (°C)")
    axs[4].legend(fontsize=8)
    axs[4].grid(True, alpha=0.3)
    
    # [5] Control Signals
    axs[5].plot(t_hr, log["u_BES"], label="BES", linewidth=1.5)
    axs[5].plot(t_hr, log["u_TES"], label="TES", linewidth=1.5)
    axs[5].axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    axs[5].set_title("Control Signals")
    axs[5].set_ylabel("Signal (-1 to 1)")
    axs[5].set_ylim(-1.2, 1.2)
    axs[5].legend(fontsize=8)
    axs[5].grid(True, alpha=0.3)
    
    # [6] HVAC Saving
    P_HVAC_saved = log["P_HVAC_baseline_kw"] - log["P_HVAC_kw"]
    axs[6].plot(t_hr, P_HVAC_saved, linewidth=1.5, color='orange')
    axs[6].fill_between(t_hr, 0, np.maximum(P_HVAC_saved, 0), alpha=0.3, color='orange')
    axs[6].set_title("HVAC Power Saved")
    axs[6].set_ylabel("Power (kW)")
    axs[6].grid(True, alpha=0.3)
    
    # [7] KPI Summary
    axs[7].axis('off')
    kpi_text = f"""
    KEY PERFORMANCE INDICATORS
    ────────────────────────────
    Peak Shaving (PSR):  {psr:.2f}%
    Baseline Peak:       {peak_baseline:.1f} kW
    Optimized Peak:      {peak_optimized:.1f} kW
    
    Avg SOC: BES {np.mean(outputs["socBES"]):.0%} | TES {np.mean(outputs["socTES"]):.0%}
    
    Max Temp: {np.max(log["T_server_C"]):.0f}°C
    
    Control Method:
    ─ Softmax Coordination
    ─ Overload-duration tracking
    ─ Off-peak charging (23-12)
    ─ Peak discharging (12-23)
    """
    axs[7].text(0.05, 0.95, kpi_text, transform=axs[7].transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    for ax in axs:
        ax.set_xlabel("Time (hours)")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        outputs, log, grid_limit_kw, step_size = run_fmu_simulation()
        plot_results(outputs, log, grid_limit_kw, step_size)
        print("\n✓ Simulation completed successfully!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()




