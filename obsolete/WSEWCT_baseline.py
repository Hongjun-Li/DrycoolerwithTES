# ASHRAE26_FMU_Latest_0TES_0FMU_0WSEWCT_0py.fmu
import numpy as np
import matplotlib.pyplot as plt
from pyfmi import load_fmu
from tqdm import tqdm

# ================= 配置 =================
# 请替换为你实际导出的包含数据中心与内置环境的 FMU 文件名
FMU_NAME = "ASHRAE26_FMU_Latest_0TES_0FMU_0WSEWCT_0py.fmu" 


START_TIME = 0.0
# 你提到模拟了21天，这里设置为 21 天 (21 * 24 * 3600 秒)
STOP_TIME = 86400.0 * 7 
# 步长设为 900 秒 (15分钟)，对齐电力现货市场调度频次
STEP_SIZE = 900.0          

def run_tes_control_test():
    print(f"Loading FMU: {FMU_NAME}...")
    try:
        # log_level=3 打印较少日志，避免刷屏
        model = load_fmu(FMU_NAME, log_level=3) 
        model.initialize(start_time=START_TIME, stop_time=STOP_TIME)
    except Exception as e: 
        print(f"Failed to load FMU: {e}")
        return

    # ================= 变量映射 =================
    # 唯一控制输入
    input_tes = "sigTES" 

    # 对应截图右侧的所有输出节点
    output_keys = [
        "Tamb",       # 环境温度 (K)
        "socTES",     # 蓄冷罐 SOC (0~1)
        "yTserver",   # 服务器温度 (K)
        "yTCHWret",   # 冷冻水回水温度 (K)
        "yTCDUSup",   # CDU 供水温度 (K)
        "yPIT",       # IT 负载功率 (W)
        "yPHVAC"      # HVAC 总能耗 (W)
    ]

    # 准备存储结果的数组
    n_steps = int((STOP_TIME - START_TIME) / STEP_SIZE)
    time_array = np.linspace(START_TIME, STOP_TIME, n_steps)
    
    results = {k: np.zeros(n_steps) for k in output_keys}
    inputs_log = {
        "sigTES": np.zeros(n_steps),
        "mock_price": np.zeros(n_steps) # 记录虚拟电价用于分析
    }

    current_time = START_TIME
    
    print("Running TES Economic Dispatch Simulation...")
    for i in tqdm(range(n_steps)):
        
        # ================== 1. 读取当前状态 ==================
        # 获取上一时刻的状态作为决策依据
        current_soc = model.get("socTES")[0]
        # 【关键修正】：将开尔文转换为摄氏度，用于逻辑判断
        current_t_server_C = model.get("yTserver")[0] - 273.15 
        
        # ================== 2. 模拟外部环境 (电价) ==================
        # 计算当前是每天的第几个小时 (0~23.99)
        hour_of_day = (current_time / 3600.0) % 24
        
        # 简单模拟峰谷电价
        if 18 <= hour_of_day < 22:
            price = 1.0  # 高峰电价 (傍晚到夜间电网紧张)
        elif 0 <= hour_of_day < 8:
            price = 0.2  # 低谷电价 (凌晨风光大发或负荷低)
        else:
            price = 0.5  # 平段电价
            
        inputs_log["mock_price"][i] = price

        # ================== 3. 核心控制逻辑 ==================
        u_tes = 0.0 # 默认静置
        
        # 【关键修正】：提高安全红线到 65°C。
        # 数据中心服务器排风/芯片温度在 50°C-60°C 是正常的。
        SERVER_TEMP_LIMIT = 65.0 
        
        # 优先级1：绝对安全约束
        if current_t_server_C > SERVER_TEMP_LIMIT: 
            # 触发安全保护，强制停止主动套利，由系统底层保障散热
            u_tes = 0.0 
        
        # 优先级2：经济套利策略
        else:
            if price == 1.0 and current_soc > 0.1:
                # 电价昂贵且有冷量 -> 放冷 (-1.0)，降低 HVAC 瞬时功率
                u_tes = -1.0
            elif price == 0.2 and current_soc < 0.9:
                # 电价便宜且有空间 -> 蓄冷 (1.0)，增加瞬时功率存积冷量
                u_tes = 1.0
            else:
                # 平价或条件不满足 -> 保持静置 (0.0)
                u_tes = 0.0

        # ================== 4. 写入输入并执行 ==================
        model.set(input_tes, u_tes)

        try:
            # 执行 15 分钟的仿真步
            model.do_step(current_t=current_time, step_size=STEP_SIZE)
        except Exception as e:
            print(f"Crash at t={current_time}: {e}")
            break

        # ================== 5. 记录输出数据 ==================
        vals = model.get(output_keys)
        for k, key in enumerate(output_keys):
            # 【关键修正】：如果是温度变量，在存入结果时直接转换为摄氏度，方便画图
            if key in ["yTserver", "yTCHWret", "yTCDUSup", "Tamb"]:
                results[key][i] = vals[k] - 273.15
            else:
                results[key][i] = vals[k]
        
        inputs_log["sigTES"][i] = u_tes
        current_time += STEP_SIZE

    # ================== 绘图检查 ==================
    plot_results(time_array, results, inputs_log)


def plot_results(time_arr, res, inp):
    # 将 X 轴转换为天数，21天的数据用小时看会太密
    t_days = time_arr / (3600.0 * 24.0)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # ---------------- 图1: 功率与电价关系 ----------------
    ax1 = axes[0]
    ax1.plot(t_days, res["yPIT"], 'k-', label="IT Load (yPIT)")
    ax1.plot(t_days, res["yPHVAC"], 'g-', alpha=0.8, label="HVAC Power (yPHVAC)")
    ax1.set_ylabel("Power [W]")
    
    # 用右轴画电价
    ax1_r = ax1.twinx()
    ax1_r.fill_between(t_days, inp["mock_price"], color='purple', alpha=0.1, step='post')
    ax1_r.step(t_days, inp["mock_price"], 'purple', linestyle=':', label="Electricity Price", where='post')
    ax1_r.set_ylabel("Price")
    ax1.set_title("1. Power Consumption vs Electricity Price")
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_r.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    ax1.grid(True, alpha=0.3)
    
    # ---------------- 图2: 蓄冷系统响应 ----------------
    ax2 = axes[1]
    ax2.plot(t_days, res["socTES"], 'orange', linestyle='-', linewidth=2, label="TES SOC")
    ax2.set_ylabel("SOC (0~1)")
    ax2.set_ylim(-0.1, 1.1) # 固定 SOC 轴的范围以便观察
    
    ax2_r = ax2.twinx()
    ax2_r.step(t_days, inp["sigTES"], 'r-', alpha=0.7, label="Input: sigTES", where='post')
    ax2_r.set_ylabel("Signal (-1: Discharge, 1: Charge)")
    ax2_r.set_ylim(-1.5, 1.5)
    ax2.set_title("2. TES Signal and State of Charge")
    
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")
    ax2.grid(True, alpha=0.3)

    # ---------------- 图3: 温度监控 (已转换为摄氏度) ----------------
    ax3 = axes[2]
    ax3.plot(t_days, res["yTserver"], 'r-', label="Server Temp (yTserver)")
    ax3.plot(t_days, res["yTCHWret"], 'b-', label="CHW Return Temp (yTCHWret)")
    ax3.plot(t_days, res["yTCDUSup"], 'c--', label="CDU Supply Temp (yTCDUSup)")
    ax3.plot(t_days, res["Tamb"], 'k:', alpha=0.5, label="Ambient Temp (Tamb)")
    
    # 增加一条安全红线辅助线
    ax3.axhline(y=65.0, color='red', linestyle='--', alpha=0.3, label="Safety Limit (65C)")
    
    ax3.set_ylabel("Temperature [°C]")
    ax3.set_xlabel("Time [Days]")
    ax3.set_title("3. System Temperatures (°C)")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_tes_control_test()