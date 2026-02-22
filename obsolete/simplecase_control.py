import numpy as np
import matplotlib.pyplot as plt
from pyfmi import load_fmu
from tqdm import tqdm

# ================= 配置 =================
# 请替换为你的FMU实际文件名
FMU_NAME = "ASHRAE26_FMU_Latest_0TES_0FMU.fmu"

# 模拟3天，看能不能复现那个周期性的Load
START_TIME = 0.0
STOP_TIME = 86400.0 * 3.0 
STEP_SIZE = 60.0

def run_basic_test():
    print(f"Loading FMU: {FMU_NAME}...")
    try:
        model = load_fmu(FMU_NAME, log_level=3)
        model.initialize(start_time=START_TIME, stop_time=STOP_TIME)
    except Exception as e:
        print(f"Failed to load FMU: {e}")
        return

    # 输入变量名 (根据你的截图推测)
    input_bes = "sigBES"
    input_tes = "sigTES" 
    input_set = "sigsetpoint"

    # 输出变量名
    output_keys = [
        "yPIT",       # IT Load
        "yPHVAC",     # HVAC Power
        "yPBES",      # BES Power (Actual)
        "yPTES",      # TES Power (Actual)
        "socBES",     # BES SOC
        "socTES",     # TES SOC
        "yTserver",   # Server Temp
        "yTCHWret"    # Return Water Temp
    ]

    # 准备存储结果
    n_steps = int((STOP_TIME - START_TIME) / STEP_SIZE)
    time_array = np.linspace(START_TIME, STOP_TIME, n_steps)
    results = {k: np.zeros(n_steps) for k in output_keys}
    inputs_log = {
        "sigBES": np.zeros(n_steps),
        "sigTES": np.zeros(n_steps),
        "setpoint": np.zeros(n_steps)
    }

    current_time = START_TIME
    
    print("Running basic step test...")
    for i in tqdm(range(n_steps)):
        
        # ================== 简单的阶跃控制逻辑 ==================
        # 1. 默认值
        u_bes = 0.0
        u_tes = 0.0
        u_set = 273.15 + 24.0  # 24°C (Kelvin)

        # 2. 按时间段测试不同设备
        hour = (current_time / 3600.0)
        
        # [0-12h]   待机观察：看 yPIT 是否自然波动
        if 0 <= hour < 12:
            pass 

        # [12-18h]  BES 充电测试 (sigBES = 1.0)
        elif 12 <= hour < 18:
            u_bes = 1.0

        # [18-24h]  BES 放电测试 (sigBES = -1.0)
        elif 18 <= hour < 24:
            u_bes = -1.0

        # [24-30h]  TES 蓄冷测试 (sigTES = 1.0)
        elif 24 <= hour < 30:
            u_tes = 1.0
        
        # [30-36h]  TES 放冷测试 (sigTES = -1.0)
        elif 30 <= hour < 36:
            u_tes = -1.0

        # [36-48h]  提高设定点测试 (28°C) -> 应该降低 HVAC 功率
        elif 36 <= hour < 48:
            u_set = 273.15 + 28.0
            
        # [48h+]    恢复默认
        else:
            pass

        # ================== 写入输入 ==================
        model.set(input_bes, u_bes)
        model.set(input_tes, u_tes)
        model.set(input_set, u_set)

        # ================== 执行一步 ==================
        try:
            model.do_step(current_t=current_time, step_size=STEP_SIZE)
        except Exception as e:
            print(f"Crash at t={current_time}: {e}")
            break

        # ================== 记录数据 ==================
        vals = model.get(output_keys)
        for k, key in enumerate(output_keys):
            results[key][i] = vals[k]
        
        inputs_log["sigBES"][i] = u_bes
        inputs_log["sigTES"][i] = u_tes
        inputs_log["setpoint"][i] = u_set - 273.15 # 存摄氏度方便看

        current_time += STEP_SIZE

    # ================== 绘图检查 ==================
    plot_basic_test(time_array, results, inputs_log)

def plot_basic_test(time_arr, res, inp):
    t_hours = time_arr / 3600.0
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    # 图1: 关键的 Load 检查 (IT + HVAC)
    ax1 = axes[0]
    ax1.plot(t_hours, res["yPIT"], 'k-', label="IT Load (yPIT)")
    ax1.plot(t_hours, res["yPHVAC"], 'g-', alpha=0.6, label="HVAC Power")
    ax1.set_ylabel("Power [W]")
    ax1.set_title("1. Load Verification (Check this matches your reference image)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    
    # 图2: BES 测试
    ax2 = axes[1]
    ax2.plot(t_hours, res["yPBES"], 'b-', label="BES Power (yPBES)")
    ax2_r = ax2.twinx()
    ax2_r.plot(t_hours, res["socBES"], 'b--', label="BES SOC")
    ax2_r.step(t_hours, inp["sigBES"], 'r:', label="Input: sigBES", where='post')
    
    ax2.set_ylabel("Power [W]")
    ax2_r.set_ylabel("SOC / Signal")
    ax2.set_title("2. Battery Response (sigBES -> Power & SOC)")
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")
    ax2.grid(True, alpha=0.3)

    # 图3: TES 测试
    ax3 = axes[2]
    ax3.plot(t_hours, res["yPTES"], 'orange', label="TES Power (yPTES)")
    ax3_r = ax3.twinx()
    ax3_r.plot(t_hours, res["socTES"], 'orange', linestyle='--', label="TES SOC")
    ax3_r.step(t_hours, inp["sigTES"], 'r:', label="Input: sigTES", where='post')
    
    ax3.set_ylabel("Power [W]")
    ax3_r.set_ylabel("SOC / Signal")
    ax3.set_title("3. TES Response (sigTES -> Power & SOC)")
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_r.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc="upper right")
    ax3.grid(True, alpha=0.3)

    # 图4: Setpoint & Temperature
    ax4 = axes[3]
    ax4.plot(t_hours, res["yTserver"] - 273.15, 'k-', label="Server Temp")
    ax4.step(t_hours, inp["setpoint"], 'm--', label="Setpoint (Input)")
    ax4.set_ylabel("Temp [C]")
    ax4.set_xlabel("Time [Hours]")
    ax4.set_title("4. Thermal Response")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_basic_test()