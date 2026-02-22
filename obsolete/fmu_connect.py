from pyfmi import load_fmu
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

def run_simulation():
    # ================= 配置区域 =================
    fmu_path = "ASHRAE26_FMU_Latest_0TES_0FMU_0test.fmu"  # 请确保 FMU 文件名一致
    
    # 仿真参数
    # 建议先跑 1 天 (86400s) 测试，如果没问题再跑 7 天 (604800s)
    start_time = 0.0
    stop_time = 86400.0 * 7  
    step_size = 60.0  # 步长 60秒。如果报错卡住，请改为 1.0 或 10.0
    
    # 定义变量名 (与 Modelica 代码一致)
    input_var_name = 'ysetpoint'
    
    # 你指定的控制判断变量
    control_trigger_var = 'yTES' 
    
    # 需要记录的输出变量
    output_keys = [
        'socTES',       # TES 荷电状态
        'yTES',         # 新增加的 TES 控制信号/模式
        'yTserver',     # 服务器温度
        'yTCDUSup',     # CDU 供水温度
        'yTCHWret',     # 冷冻水回水温度
        'yPIT',         # IT 功率
        'yPHVAC',       # HVAC 功率
        'socBES'        # 电池 SOC
    ]
    # ===========================================

    print(f"1. 正在加载 FMU: {fmu_path} ...")
    try:
        # log_level=3 只有警告和错误，减少打印提高速度
        model = load_fmu(fmu_path, log_level=3)
        print("   FMU 加载成功。")
    except Exception as e:
        print(f"!!! FMU 加载失败: {e}")
        return

    # --- 检查变量连接 (Connect Check) ---
    print("2. 检查关键变量连接...")
    all_vars = model.get_model_variables()
    
    if control_trigger_var in all_vars:
        print(f"   [OK] 找到输出变量 '{control_trigger_var}'")
    else:
        print(f"   [Error] 找不到变量 '{control_trigger_var}'。请检查 FMU 是否重新导出。")
        # 打印可能的变量名帮助调试
        candidates = [k for k in all_vars.keys() if 'TES' in k]
        print(f"   提示: 包含 'TES' 的变量有: {candidates[:5]}...")
        return

    if input_var_name in all_vars:
        print(f"   [OK] 找到输入变量 '{input_var_name}'")
    else:
        print(f"   [Error] 找不到输入变量 '{input_var_name}'")
        return

    # --- 初始化 ---
    print(f"3. 初始化模型 (Stop Time: {stop_time}s)...")
    try:
        model.initialize(start_time=start_time, stop_time=stop_time)
        print("   初始化完成。")
    except Exception as e:
        print(f"!!! 初始化失败: {e}")
        return

    # --- 准备存储数组 ---
    n_steps = int((stop_time - start_time) / step_size)
    
    # 使用 numpy 数组比 list append 更快
    results = {key: np.zeros(n_steps) for key in output_keys}
    results['time'] = np.zeros(n_steps)
    results['ysetpoint'] = np.zeros(n_steps)

    current_time = start_time
    
    # --- 仿真循环 ---
    print(f"4. 开始仿真 (Step Size: {step_size}s)...")
    
    # tqdm 显示进度条
    for i in tqdm(range(n_steps), desc="Running", unit="step"):
        
        # A. 获取当前状态 (用于决定下一步的输入)
        try:
            # 获取 yTES 的值
            val_trigger = model.get(control_trigger_var)[0]
        except:
            val_trigger = 0.0 # 默认值，防止第一步读取失败

        # B. 你的控制逻辑
        # "input的值给定一个273.15+50，如果yTES小于0，则273.15+52"
        if val_trigger < 0:
            u_val = 273.15 + 52.0
        else:
            u_val = 273.15 + 50.0
        
        # 设置输入
        model.set(input_var_name, u_val)

        # C. 执行一步仿真
        try:
            model.do_step(current_t=current_time, step_size=step_size)
        except Exception as e:
            print(f"\n[Crash] 仿真在 t={current_time} 处崩溃: {e}")
            # 截断数据以便绘图分析崩溃前的情况
            for k in results:
                results[k] = results[k][:i]
            break

        # D. 记录数据
        results['time'][i] = current_time
        results['ysetpoint'][i] = u_val - 273.15 # 存摄氏度方便看
        
        # 批量读取输出
        vals = model.get(output_keys)
        for idx, key in enumerate(output_keys):
            results[key][i] = vals[idx]

        current_time += step_size

    print("仿真结束。正在绘图...")

    # --- 绘图 ---
    if len(results['time']) == 0:
        return

    # 将时间转换为小时
    time_hr = results['time'] / 3600.0
    
    # 创建 4 个子图
    fig, axes = plt.subplots(4, 1, figsize=(10, 6), sharex=True)
    
    # 1. 输入控制与触发信号
    ax1 = axes[0]
    ax1.step(time_hr, results['yTES'], label='yTES', where='post')
    ax1.plot(time_hr, results['socTES'], label='SOC TES', color='tab:orange', linewidth=2)
    ax1.set_ylabel('yTES & socTES')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    
    # 2. SOC 状态
    ax2 = axes[1]
    
    ax2.plot(time_hr, results['socBES'], label='SOC Battery', color='tab:green', linestyle=':')
    ax2.set_ylabel('SOC [0-1]')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    # 3. 温度
    ax3 = axes[2]
    ax3.plot(time_hr, results['yTserver'] - 273.15, label='Server Temp')
    ax3.plot(time_hr, results['yTCDUSup'] - 273.15, label='CDU Supply')
    ax3.plot(time_hr, results['yTCHWret'] - 273.15, label='CHW Return')
    ax3.set_ylabel('Temperature [°C]')
    ax3.legend(loc='upper right')
    ax3.grid(True)

    # 4. 功率
    ax4 = axes[3]
    ax4.plot(time_hr, results['yPHVAC'], label='HVAC Power')
    ax4.plot(time_hr, results['yPIT'], label='IT Power')
    ax4.set_ylabel('Power [W]')
    ax4.set_xlabel('Time [hours]')
    ax4.legend(loc='upper right')
    ax4.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()