import numpy as np
import matplotlib.pyplot as plt
from pyfmi import load_fmu
from tqdm import tqdm

# ================= 配置 =================
FMU_NAME = "ASHRAE26_FMU_Latest_0TES_0FMU_0WSEWCT_0py.fmu" 

# 【核心修改】：将起始时间挪到 7 月中旬 (第 195 天)
START_DAY = 195.0
START_TIME = START_DAY * 86400.0
# 结束时间为起始时间加上 7 天
STOP_TIME = START_TIME + (86400.0 * 7.0) 
STEP_SIZE = 900.0          # 15分钟步长

# 设定点配置 (开尔文 K)
SP_DEFAULT = 300.15  # 默认 27°C
SP_CHARGE  = 297.15  # 蓄冷时降温到 24°C (逼迫风机在半夜全速造冷)
SP_DISCHARGE = 305.15  # 放冷时提高到 32°C (让风机/水泵大幅降频休息)

def run_simulation(strategy="baseline"):
    print(f"\n--- 启动仿真: [{strategy.upper()}] 模式 ---")
    model = load_fmu(FMU_NAME, log_level=4) 
    model.initialize(start_time=START_TIME, stop_time=STOP_TIME)

    output_keys = ["Tamb", "socTES", "yTserver", "yTCHWret", "yTCDUSup", "yPIT", "yPHVAC"]
    n_steps = int((STOP_TIME - START_TIME) / STEP_SIZE)
    time_array = np.linspace(START_TIME, STOP_TIME, n_steps)
    
    results = {k: np.zeros(n_steps) for k in output_keys}
    inputs_log = {
        "sigTES": np.zeros(n_steps), 
        "sigsetpoint": np.zeros(n_steps), 
        "mock_price": np.zeros(n_steps)
    }
    
    cost_log = np.zeros(n_steps)
    current_time = START_TIME
    
    for i in tqdm(range(n_steps), desc=f"Simulating {strategy}"):
        current_soc = model.get("socTES")[0]
        current_t_server_C = model.get("yTserver")[0] - 273.15 
        current_Tamb_C = model.get("Tamb")[0] - 273.15 
        
        # 1. 获取电价
        hour_of_day = (current_time / 3600.0) % 24
        if 18 <= hour_of_day < 22:
            price = 1.0  
        elif 0 <= hour_of_day < 8:
            price = 0.2  
        else:
            price = 0.5  
            
        inputs_log["mock_price"][i] = price

        # 2. 核心联合控制逻辑 (TES + Setpoint)
        u_tes = 0.0 
        u_sp = SP_DEFAULT 
        
        if strategy == "baseline":
            u_tes = 0.0
            u_sp = SP_DEFAULT
            
        elif strategy == "rbc":
            SERVER_TEMP_LIMIT = 65.0   # 绝对死线
            SERVER_TEMP_WARNING = 60.0 # 预警线
            
            if current_t_server_C > SERVER_TEMP_LIMIT: 
                u_tes = 0.0 
                u_sp = SP_DEFAULT 
                
            else:
                if price == 1.0 and current_soc > 0.1:
                    u_tes = -1.0
                    if current_t_server_C > SERVER_TEMP_WARNING:
                        u_sp = SP_DEFAULT 
                    else:
                        u_sp = SP_DISCHARGE 
                        
                elif price == 0.2 and current_soc < 0.9 and current_Tamb_C < 25.0:
                    # 夏天夜里没那么凉快，把蓄冷的室外温度门槛适当放宽到 25 度
                    u_tes = 1.0  
                    u_sp = SP_CHARGE 
                    
                else:
                    u_tes = 0.0
                    u_sp = SP_DEFAULT

        # 下发两个信号
        model.set("sigTES", u_tes)
        model.set("sigsetpoint", u_sp)
        
        # 执行仿真
        model.do_step(current_t=current_time, step_size=STEP_SIZE)

        # 3. 记录数据
        vals = model.get(output_keys)
        for k, key in enumerate(output_keys):
            if key in ["yTserver", "yTCHWret", "yTCDUSup", "Tamb"]:
                results[key][i] = vals[k] - 273.15
            else:
                results[key][i] = vals[k]
        
        inputs_log["sigTES"][i] = u_tes
        inputs_log["sigsetpoint"][i] = u_sp - 273.15 
        
        # 4. 计算当前 15 分钟的电费
        total_power_kw = (results["yPIT"][i] + results["yPHVAC"][i]) / 1000.0
        energy_kwh = total_power_kw * (STEP_SIZE / 3600.0)
        cost_log[i] = energy_kwh * price

        current_time += STEP_SIZE
        
    return time_array, results, inputs_log, cost_log


def run_comparison():
    time_arr, res_base, inp_base, cost_base = run_simulation("baseline")
    _, res_rbc, inp_rbc, cost_rbc = run_simulation("rbc")

    total_cost_base = np.sum(cost_base)
    total_cost_rbc = np.sum(cost_rbc)
    savings = total_cost_base - total_cost_rbc
    savings_percent = (savings / total_cost_base) * 100 if total_cost_base > 0 else 0

    print("\n" + "="*45)
    print("💰 盛夏7天：联合调度电费核算报告 (TES + Setpoint) 💰")
    print("="*45)
    print(f"不使用 TES (Baseline): {total_cost_base:,.2f} 元")
    print(f"联合套利 (RBC):         {total_cost_rbc:,.2f} 元")
    print("-" * 45)
    if savings > 0:
        print(f"✅ 成功节约电费: {savings:,.2f} 元 ({savings_percent:.2f}%)")
    else:
        print(f"❌ 亏损或无节约: {savings:,.2f} 元")
    print("="*45 + "\n")

    # ================= 绘图 =================
    # 【画图修改】：将 X 轴平移，使其依然显示第 0~7 天，方便查看
    t_days = (time_arr - START_TIME) / (3600.0 * 24.0)
    
    cumulative_cost_base = np.cumsum(cost_base)
    cumulative_cost_rbc = np.cumsum(cost_rbc)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # 图1：累计电费
    ax1 = axes[0]
    ax1.plot(t_days, cumulative_cost_base, 'k--', linewidth=2, label=f"Baseline Cost")
    ax1.plot(t_days, cumulative_cost_rbc, 'g-', linewidth=2, label=f"RBC Cost")
    ax1.fill_between(t_days, cumulative_cost_base, cumulative_cost_rbc, color='green', alpha=0.1)
    ax1.set_ylabel("Cumulative Cost (¥)")
    ax1.set_title("1. Financial Comparison (Summer Week)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.4)

    # 图2：HVAC 功率对比
    ax2 = axes[1]
    ax2.plot(t_days, res_base["yPHVAC"], 'k--', alpha=0.5, label="HVAC Power - Baseline")
    ax2.plot(t_days, res_rbc["yPHVAC"], 'g-', alpha=0.8, label="HVAC Power - With TES & Setpoint")
    
    ax2_r = ax2.twinx()
    ax2_r.fill_between(t_days, inp_rbc["mock_price"], color='purple', alpha=0.05, step='post')
    ax2_r.step(t_days, inp_rbc["mock_price"], 'purple', linestyle=':', alpha=0.5, label="Price", where='post')
    ax2_r.set_ylabel("Price")
    
    ax2.set_ylabel("HVAC Power [W]")
    ax2.set_title("2. HVAC Power Load Profile")
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")
    ax2.grid(True, alpha=0.4)

    # 图3：控制信号与设定点
    ax3 = axes[2]
    ax3.plot(t_days, res_rbc["yTserver"], 'r-', alpha=0.7, label="Actual Server Temp (°C)")
    # 偷偷看一眼室外温度，确认是不是夏天
    ax3.plot(t_days, res_rbc["Tamb"], 'gray', linestyle=':', alpha=0.5, label="Ambient Temp (°C)")
    
    ax3.step(t_days, inp_rbc["sigsetpoint"], 'b--', linewidth=2, label="Dynamic Setpoint (°C)", where='post')
    ax3.axhline(y=65.0, color='red', linestyle=':', label="Hard Safety Limit (65°C)")
    ax3.axhline(y=60.0, color='orange', linestyle=':', label="Warning Limit (60°C)")
    ax3.set_ylabel("Temperature (°C)")
    
    ax3_r = ax3.twinx()
    ax3_r.step(t_days, inp_rbc["sigTES"], 'm-', alpha=0.5, label="TES Signal", where='post')
    ax3_r.set_ylabel("TES Signal")
    ax3_r.set_ylim(-1.5, 1.5)
    
    ax3.set_xlabel("Time [Relative Days]")
    ax3.set_title("3. System Temperatures & Control Signals")
    
    lines, labels = ax3.get_legend_handles_labels()
    lines3, labels3 = ax3_r.get_legend_handles_labels()
    ax3.legend(lines + lines3, labels + labels3, loc="upper left")
    ax3.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_comparison()