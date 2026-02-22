#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from pyfmi import load_fmu
from tqdm import tqdm

def run_fmu_simulation():
    """
    Simple co-simulation with Latest_TES_FMU model.
    Controls: sigBES, sigTES (range -1~1), sigsetpoint (K, around 273.15+50)
    """
    
    # ============ Configuration ============
    fmu_name = "ASHRAE26_FMU_Latest_0TES_0FMU.fmu"
    start_time = 0.0
    stop_time = 86400.0 * 2.0  # 2 days
    step_size = 300.0  # 5 minutes
    dt_hr = step_size / 3600.0
    
    # Input/Output names
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
    
    # Control parameters
    SETPOINT_TARGET_K = 273.15 + 50.0
    GRID_LIMIT_KW = 500.0
    IT_PEAK_KW = 500.0
    
    # ============ Load and Initialize FMU ============
    print(f"Loading FMU: {fmu_name}")
    model = load_fmu(fmu_name, log_level=3)
    
    model.initialize(start_time=start_time, stop_time=stop_time)
    model.set(input_names["Set"], SETPOINT_TARGET_K)
    
    n_steps = int((stop_time - start_time) / step_size)
    
    # Initialize output arrays
    outputs = {name: np.zeros(n_steps) for name in output_names}
    outputs["time"] = np.zeros(n_steps)
    
    # Logging arrays
    log = {
        "P_IT_kw": np.zeros(n_steps),
        "P_HVAC_kw": np.zeros(n_steps),
        "P_HVAC_baseline_kw": np.zeros(n_steps),  # HVAC power with fixed setpoint (50°C)
        "P_BES_kw": np.zeros(n_steps),
        "P_TES_kw": np.zeros(n_steps),
        "u_BES": np.zeros(n_steps),
        "u_TES": np.zeros(n_steps),
        "T_server_C": np.zeros(n_steps),
        "T_CHWret_C": np.zeros(n_steps),
        "T_CDUsup_C": np.zeros(n_steps),
        "P_grid_kw": np.zeros(n_steps),
        "setpoint_cmd_K": np.zeros(n_steps),
        "overflow_kw": np.zeros(n_steps),
        # Progressive control effects
        "P_setpoint_only": np.zeros(n_steps),      # Only setpoint, no BES/TES
        "P_bes_setpoint": np.zeros(n_steps),        # BES + setpoint
        "P_bes_tes_setpoint": np.zeros(n_steps),    # BES + TES + setpoint (full)
    }
    
    # Previous states
    u_bes_prev = 0.0
    u_tes_prev = 0.0
    setpoint_prev = SETPOINT_TARGET_K  # Track setpoint changes
    
    # Store setpoint and control signals for baseline comparison
    setpoint_log = []
    u_bes_log = []
    u_tes_log = []
    
    current_time = start_time
    
    print(f"Running simulation for {stop_time/86400:.1f} days with {n_steps} steps...")
    
    # ============ Main Control Loop ============
    for i in tqdm(range(n_steps)):
        # Get current outputs
        if i == 0:
            # Initialize with default values
            # Make initial SOC LOW to trigger charging
            yPIT_kw = IT_PEAK_KW * 0.95  # Higher initial load to create overflow
            yPHVAC_kw = 60.0
            socBES = 0.45  # Below charge threshold (0.58)
            socTES = 0.45  # Below charge threshold (0.55)
            Tserver_C = 35.0
        else:
            yPIT_kw = float(outputs["yPIT"][i-1]) / 1000.0
            yPHVAC_kw = float(outputs["yPHVAC"][i-1]) / 1000.0
            socBES = float(outputs["socBES"][i-1])
            socTES = float(outputs["socTES"][i-1])
            Tserver_C = float(outputs["yTserver"][i-1]) - 273.15
        
        P_base_kw = yPIT_kw + yPHVAC_kw
        
        # ========== Direct Control with SOC Limits ==========
        # Prevent over-discharge and over-charge
        
        P_grid_no_storage = P_base_kw
        overflow = max(0.0, P_grid_no_storage - GRID_LIMIT_KW)
        
        # Get current hour for time-based control
        hour = (current_time / 3600.0) % 24.0
        is_peak_hour = 12 <= hour < 23  # High price: 12:00-23:00
        is_off_peak = 23 <= hour or hour < 12  # Low price: 23:00-12:00
        is_high_price = is_peak_hour  # High price during peak hours
        
        # ======= SOC Limits =======
        BES_SOC_MIN, BES_SOC_MAX = 0.1, 0.9
        TES_SOC_MIN, TES_SOC_MAX = 0.2, 0.8
        
        # ======= BES Control: Discharge Priority, NO CHARGING at High Price =======
        # High price = complete BAN on charging (only discharge or 0)
        
        u_bes_cmd = 0.0
        
        # Discharge when overflow is high
        if overflow > 50.0 and socBES > BES_SOC_MIN + 0.05:
            u_bes_cmd = -0.7  # Aggressive discharge
        # LOW price periods ONLY: Limited charging (off-peak only 23-12)
        elif is_off_peak and socBES < BES_SOC_MAX - 0.15:
            u_bes_cmd = 0.2   # Charge ONLY during off-peak (23-12)
        else:
            u_bes_cmd = 0.0   # Neutral
        
        # FORCE NO CHARGING during high price: clamp to <= 0
        if is_high_price and u_bes_cmd > 0:
            u_bes_cmd = 0.0  # Force to zero if charging during high price
        
        # ======= TES Control: Discharge Priority, NO CHARGING at High Price =======
        # Same rule: BAN charging at high price, only discharge or 0
        
        u_tes_cmd = 0.0
        
        # Discharge when high overflow during peak
        if overflow > 80.0 and is_peak_hour and socTES > TES_SOC_MIN + 0.05:
            u_tes_cmd = -0.6
        # LOW price ONLY: Conservative charging during off-peak (23-12)
        elif is_off_peak and socTES < TES_SOC_MAX - 0.15:
            u_tes_cmd = 0.2   # Charge ONLY during off-peak (23-12)
        else:
            u_tes_cmd = 0.0
        
        # FORCE NO CHARGING during high price: clamp to <= 0
        if is_high_price and u_tes_cmd > 0:
            u_tes_cmd = 0.0  # Force to zero if charging during high price
        
        # Smooth transitions
        u_bes_cmd = 0.5 * u_bes_prev + 0.5 * u_bes_cmd  # Faster, more responsive
        u_tes_cmd = 0.5 * u_tes_prev + 0.5 * u_tes_cmd  # Faster, more responsive
        
        # Clamp to valid ranges
        u_bes_cmd = float(np.clip(u_bes_cmd, -1.0, 1.0))
        u_tes_cmd = float(np.clip(u_tes_cmd, -1.0, 1.0))
        
        # ======= SETPOINT Dynamic Adjustment =======
        # Setpoint is the target Server Temperature (Tserver)
        # When grid power exceeds 500kW, can raise setpoint to reduce HVAC load
        # Base: 50°C, Max: 60°C
        
        SETPOINT_BASE_C = 50.0
        SETPOINT_MAX_C = 60.0
        SETPOINT_BASE_K = 273.15 + SETPOINT_BASE_C
        SETPOINT_MAX_K = 273.15 + SETPOINT_MAX_C
        
        # Adjust setpoint based on total grid power
        P_total_grid = P_base_kw  # Total baseline power before storage/setpoint adjustment
        
        if P_total_grid > 500.0:
            # Increase setpoint when power exceeds limit
            # Formula: +0.1°C per kW above 500kW, capped at 60°C
            dT_increase = (P_total_grid - 500.0) * 0.1  # 0.1°C per kW
            setpoint_target_C = SETPOINT_BASE_C + dT_increase
            setpoint_target_C = float(np.clip(setpoint_target_C, SETPOINT_BASE_C, SETPOINT_MAX_C))
        else:
            # Normal operation
            setpoint_target_C = SETPOINT_BASE_C
        
        setpoint_target_K = 273.15 + setpoint_target_C
        
        # Smooth transitions to avoid abrupt temperature changes
        setpoint_cmd = 0.85 * setpoint_prev + 0.15 * setpoint_target_K
        setpoint_prev = setpoint_cmd
        
        # Set FMU inputs
        model.set(input_names["BES"], u_bes_cmd)
        model.set(input_names["TES"], u_tes_cmd)
        model.set(input_names["Set"], setpoint_cmd)
        
        # Execute one step
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
        
        # Sanity check: log SOC warning if approaching limits
        if socBES < 0.05:
            print(f"\n⚠️  WARNING at step {i}: BES SOC too low ({socBES:.4f}), limiting discharge")
            print(f"   Consider reducing discharge power or increasing charging")
        if socBES > 0.95:
            print(f"\n⚠️  WARNING at step {i}: BES SOC too high ({socBES:.4f}), limiting charge")
        if socTES < 0.15:
            print(f"\n⚠️  WARNING at step {i}: TES SOC too low ({socTES:.4f}), limiting discharge")
        if socTES > 0.95:
            print(f"\n⚠️  WARNING at step {i}: TES SOC too high ({socTES:.4f}), limiting charge")
        log["u_BES"][i] = u_bes_cmd
        log["u_TES"][i] = u_tes_cmd
        log["T_server_C"][i] = float(outputs["yTserver"][i]) - 273.15
        log["T_CHWret_C"][i] = float(outputs["yTCHWret"][i]) - 273.15
        log["T_CDUsup_C"][i] = float(outputs["yTCDUSup"][i]) - 273.15
        # Note: P_BES and P_TES are negative when discharging, positive when charging
        # So we add them directly (not subtract)
        P_grid_kw = yPIT_kw + yPHVAC_kw + log["P_BES_kw"][i] + log["P_TES_kw"][i]
        log["P_grid_kw"][i] = P_grid_kw
        log["setpoint_cmd_K"][i] = setpoint_cmd
        log["overflow_kw"][i] = overflow
        
        # Calculate progressive control effects for visualization
        log["P_setpoint_only"][i] = yPIT_kw + yPHVAC_kw  # Baseline with setpoint (no BES/TES)
        log["P_bes_setpoint"][i] = yPIT_kw + yPHVAC_kw + log["P_BES_kw"][i]  # Add BES effect
        log["P_bes_tes_setpoint"][i] = P_grid_kw  # Full optimization (BES+TES+setpoint)
        
        # Update previous states
        u_bes_prev = u_bes_cmd
        u_tes_prev = u_tes_cmd
        current_time += step_size
        
        # Store for baseline comparison
        setpoint_log.append(setpoint_cmd)
        u_bes_log.append(u_bes_cmd)
        u_tes_log.append(u_tes_cmd)
    
    # ============ Run Baseline Simulation (Fixed Setpoint = 50°C) ============
    # Use the same BES/TES control signals but with fixed setpoint at 50°C
    print("\nComputing baseline HVAC power (fixed setpoint = 50°C)...")
    model_baseline = load_fmu(fmu_name, log_level=3)
    model_baseline.initialize(start_time=start_time, stop_time=stop_time)
    
    SETPOINT_BASELINE_K = 273.15 + 50.0
    model_baseline.set(input_names["Set"], SETPOINT_BASELINE_K)
    
    current_time = start_time
    for i in tqdm(range(n_steps), desc="Baseline simulation"):
        # Use the same BES/TES controls as the actual simulation
        model_baseline.set(input_names["BES"], u_bes_log[i])
        model_baseline.set(input_names["TES"], u_tes_log[i])
        model_baseline.set(input_names["Set"], SETPOINT_BASELINE_K)
        
        model_baseline.do_step(current_t=current_time, step_size=step_size)
        
        vals_baseline = model_baseline.get(output_names)
        yPHVAC_baseline_kw = float(vals_baseline[output_names.index("yPHVAC")]) / 1000.0
        log["P_HVAC_baseline_kw"][i] = yPHVAC_baseline_kw
        
        current_time += step_size
    
    return outputs, log, SETPOINT_TARGET_K, GRID_LIMIT_KW, step_size


def plot_results(outputs, log, setpoint_k, grid_limit_kw, step_size):
    """
    Plot all results in a single figure with subplots.
    """
    t_hr = outputs["time"] / 3600.0
    dt_hr = step_size / 3600.0
    
    # Compute baseline vs optimized
    P_baseline = log["P_IT_kw"] + log["P_HVAC_kw"]
    P_optimized = log["P_bes_tes_setpoint"]  # BES + TES + Setpoint optimization
    
    peak_baseline = np.max(P_baseline)
    peak_optimized = np.max(P_optimized)
    psr = (peak_baseline - peak_optimized) / peak_baseline * 100.0
    
    # Create figure with 8 subplots (2x4 grid)
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs = axs.flatten()
    
    # [0] Grid Power & Baseline
    axs[0].plot(t_hr, P_baseline, label="Baseline (No Control)", linewidth=1, color='gray', linestyle='--')
    axs[0].plot(t_hr, log["P_setpoint_only"], label="+ Setpoint Only", linewidth=1, color='orange')
    axs[0].plot(t_hr, log["P_bes_setpoint"], label="+ BES + Setpoint", linewidth=1, color='green')
    axs[0].plot(t_hr, log["P_bes_tes_setpoint"], label="+ BES + TES + Setpoint", linewidth=1, color='red')
    axs[0].axhline(grid_limit_kw, color='black', linestyle=':', linewidth=1, label=f"Grid Limit ({grid_limit_kw:.0f} kW)")
    axs[0].set_title("Grid Power: Progressive Control Effects", fontsize=11)
    axs[0].set_ylabel("Power (kW)")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(fontsize=9, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=2, frameon=True)
    axs[0].set_xlabel("Time (hours)")
    
    # [1] Storage Power Output
    axs[1].plot(t_hr, log["P_BES_kw"], label="BES Power", linewidth=1.5)
    axs[1].plot(t_hr, log["P_TES_kw"], label="TES Power", linewidth=1.5)
    axs[1].set_title("Storage Power Output")
    axs[1].set_ylabel("Power (kW)")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(fontsize=9)
    
    # [2] Storage SOC
    axs[2].plot(t_hr, outputs["socBES"], label="BES SOC", linewidth=1.5)
    axs[2].plot(t_hr, outputs["socTES"], label="TES SOC", linewidth=1.5)
    axs[2].set_title("Storage State of Charge")
    axs[2].set_ylabel("SOC (0-1)")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(fontsize=9)
    
    # [3] Temperature Profile
    axs[3].plot(t_hr, log["T_server_C"], label="Server Temp", linewidth=1.5)
    axs[3].plot(t_hr, log["T_CHWret_C"], label="CHW Return", linewidth=1.5)
    axs[3].plot(t_hr, log["T_CDUsup_C"], label="CDU Supply", linewidth=1.5)
    axs[3].axhline(60, color='r', linestyle='--', label="T_safe (60°C)")
    axs[3].set_title("Temperature Profile")
    axs[3].set_ylabel("Temperature (°C)")
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(fontsize=9)
    
    # [4] HVAC Power Saving via Dynamic Setpoint Adjustment
    # Show power saved by raising setpoint to reduce HVAC load
    P_HVAC_saved = log["P_HVAC_baseline_kw"] - log["P_HVAC_kw"]
    axs[4].plot(t_hr, P_HVAC_saved, label="Power Saved (Fixed 50°C vs Dynamic)", linewidth=1, color='orange')
    axs[4].fill_between(t_hr, 0, np.maximum(P_HVAC_saved, 0), alpha=0.2, color='orange')
    axs[4].set_title("HVAC Power Saved via Dynamic Setpoint")
    axs[4].set_ylabel("Power Saved (kW)")
    axs[4].grid(True, alpha=0.3)
    axs[4].legend(fontsize=9)
    
    # [5] Control Signals (Storage Control)
    axs[5].plot(t_hr, log["u_BES"], label="sigBES (Charge -/Discharge +)", linewidth=1.5, color='blue')
    axs[5].plot(t_hr, log["u_TES"], label="sigTES (Charge -/Discharge +)", linewidth=1.5, color='green')
    axs[5].axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
    axs[5].set_title("Storage Control Strategy (BES & TES)")
    axs[5].set_ylabel("Control Signal (-1 to 1)")
    axs[5].set_ylim(-1.2, 1.2)
    axs[5].grid(True, alpha=0.3)
    axs[5].legend(fontsize=9)
    
    # [6] Cumulative Cost
    price_baseline = np.full_like(t_hr, 1.0)  # Constant price for simplicity
    cost_baseline = np.cumsum(P_baseline * dt_hr * price_baseline)
    cost_optimized = np.cumsum(P_optimized * dt_hr * price_baseline)
    
    axs[6].plot(t_hr, cost_baseline, label="Baseline Cost", linewidth=1.5)
    axs[6].plot(t_hr, cost_optimized, label="Optimized Cost", linewidth=1.5)
    savings = (cost_baseline[-1] - cost_optimized[-1]) / cost_baseline[-1] * 100.0
    axs[6].set_title(f"Cumulative Cost (Savings: {savings:.1f}%)")
    axs[6].set_ylabel("Cost (proxy)")
    axs[6].grid(True, alpha=0.3)
    axs[6].legend(fontsize=9)
    
    # [7] KPI Summary
    axs[7].axis('off')
    kpi_text = f"""
    KEY PERFORMANCE INDICATORS
    ────────────────────────────
    
    Peak Reduction (PSR):  {psr:.2f}%
    Baseline Peak:         {peak_baseline:.2f} kW
    Optimized Peak:        {peak_optimized:.2f} kW
    Grid Limit:            {grid_limit_kw:.2f} kW
    
    Avg Baseline Power:    {np.mean(P_baseline):.2f} kW
    Avg Optimized Power:   {np.mean(P_optimized):.2f} kW
    
    Avg BES SOC:           {np.mean(outputs["socBES"]):.2%}
    Avg TES SOC:           {np.mean(outputs["socTES"]):.2%}
    
    Max Server Temp:       {np.max(log["T_server_C"]):.2f}°C
    Min Server Temp:       {np.min(log["T_server_C"]):.2f}°C
    
    Setpoint Adjustment:
      Min Setpoint:        {np.min(log["setpoint_cmd_K"])-273.15:.2f}°C
      Max Setpoint:        {np.max(log["setpoint_cmd_K"])-273.15:.2f}°C
      Avg Setpoint:        {np.mean(log["setpoint_cmd_K"])-273.15:.2f}°C
    """
    axs[7].text(0.1, 0.9, kpi_text, transform=axs[7].transAxes, 
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Set common x-label
    for ax in axs:
        ax.set_xlabel("Time (hours)")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        outputs, log, setpoint_k, grid_limit_kw, step_size = run_fmu_simulation()
        plot_results(outputs, log, setpoint_k, grid_limit_kw, step_size)
        print("\n✓ Simulation completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
