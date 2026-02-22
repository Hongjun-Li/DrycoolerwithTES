import numpy as np
import matplotlib.pyplot as plt
from pyfmi import load_fmu
from tqdm import tqdm

# 1. 加载 FMU
fmu_path = 'ASHRAE26_ChillerPlant_0tes_DataCenterDryFMU.fmu'
model = load_fmu(fmu_path)

# 仿真参数（第191天到第198天）
SIM_START_DAY = 191
SIM_END_DAY = 198


if SIM_START_DAY < 1 or SIM_END_DAY < SIM_START_DAY:
    raise ValueError("SIM_START_DAY / SIM_END_DAY 设置非法")

start_time = 86400.0 * (SIM_START_DAY - 1)
stop_time = 86400.0 * SIM_END_DAY
step_size = 60.0             # 1分钟步长
steps = int((stop_time - start_time) / step_size)

# 定义变量列表
plot_vars = ['yPHVAC', 'yPIT', 'yPDCTFan', 'yPpum', 'yTdry', 'yTCWLeaTow', 'yTCDUSup', 'yTCDURet', 'ySOCtes']

def run_simulation(use_control=True):
    """运行仿真函数：可以选择开启控制或保持基准(SigTES=0)"""
    model.reset()
    model.setup_experiment(start_time=start_time)
    model.enter_initialization_mode()
    model.exit_initialization_mode()

    data = {var: [] for var in plot_vars}
    data.update({'time': [], 'sig_tes': [], 't_dry_bul': [], 'tcw_dry_set': [], 'pue': []})
    
    current_time = start_time
    sig_tes = 0.0

    desc = "仿真(有控制)" if use_control else "仿真(基准/无控制)"
    for _ in tqdm(range(steps), desc=desc):
        # --- A. 获取传感器读数 ---
        soc = model.get('ySOCtes')[0]
        try:
            t_db = model.get('weaBus.TDryBul')[0]
        except:
            t_db = 293.15

        if use_control:
            # --- B. TES 控制逻辑 (根据逻辑图 image_3b452b.png) ---
            day_sec = current_time % 86400
            is_day = 8 * 3600 <= day_sec <= 20 * 3600
            is_night = not is_day

            if sig_tes == 1.0: # 充电中
                if soc >= 0.7: sig_tes = 0.0
            elif sig_tes == -1.0: # 放电中
                if soc <= 0.05: sig_tes = 0.0
            else: # 触发充电/放电
                if is_night and soc < 0.5: sig_tes = 1.0
                elif (t_db > 273.15 + 35) or (is_day and soc > 0.65): sig_tes = -1.0
        else:
            sig_tes = 0.0 # 基准模式始终为0

        # --- C. Dry Cooler Setpoint 调节 (根据逻辑图 image_3b4ccb.png) ---
        t_base = min(43 + 273.15, max(35 + 273.15, t_db + 5.0))
        t_set = t_base - 3.0 if (sig_tes == 1.0 and is_night and use_control) else t_base

        # --- D. 执行步进 ---
        model.set('SigTES', sig_tes)
        model.set('TCWDry', t_set)
        model.do_step(current_time, step_size, True)

        # --- 数据记录 ---
        c_pit = model.get('yPIT')[0]
        c_phvac = model.get('yPHVAC')[0]
        
        data['time'].append(current_time / 86400)
        data['sig_tes'].append(sig_tes)
        data['t_dry_bul'].append(t_db - 273.15)
        data['tcw_dry_set'].append(t_set - 273.15)
        data['pue'].append((c_pit + c_phvac) / c_pit if c_pit > 0 else 1.0)
        for var in plot_vars:
            data[var].append(model.get(var)[0])
        
        current_time += step_size
    return data

# 执行两次仿真
res_base = run_simulation(use_control=False)
res_ctrl = run_simulation(use_control=True)

# 4. 绘图：2列布局
fig, axes = plt.subplots(3, 2, figsize=(12, 7), sharex=True)
plt.subplots_adjust(hspace=0.3, wspace=0.25) 

# --- 左列：电力与效能 ---
# [0,0] 绘制两条 PHVAC 曲线进行对比
axes[0,0].plot(res_ctrl['time'], res_ctrl['yPHVAC'], label='PHVAC (with TES Control)', color='red', linewidth=1.5)
axes[0,0].plot(res_base['time'], res_base['yPHVAC'], label='PHVAC (Baseline, Sig=0)', color='blue', linestyle='--', alpha=0.7)
axes[0,0].set_ylabel('Power [W]')
axes[0,0].legend(loc='upper right', fontsize='small')
axes[0,0].set_title('PHVAC Comparison: Controlled vs Baseline')

# [1,0] PUE 计算值 (显示有控制的情况)
avg_pue = np.mean(res_ctrl['pue'])
axes[1,0].plot(res_ctrl['time'], res_ctrl['pue'], color='red', label='Controlled PUE')
axes[1,0].axhline(y=avg_pue, color='darkred', linestyle='--', label=f'Avg PUE: {avg_pue:.3f}')
axes[1,0].set_ylim(1.0, 1.2) # 根据 PUE 公式修正起始点
axes[1,0].set_ylabel('PUE [-]')
axes[1,0].legend(loc='upper right')
axes[1,0].set_title('Power Usage Effectiveness (PUE)')

# [2,0] 辅机功耗对比：分别绘制风机和水泵在两种模式下的曲线
# 风机功率对比 (Fan Power)
axes[2,0].plot(res_ctrl['time'], res_ctrl['yPDCTFan'], label='Fan (Controlled)', color='#ff9999', linewidth=1.5)
axes[2,0].plot(res_base['time'], res_base['yPDCTFan'], label='Fan (Baseline)', color='#ff9999', linestyle='--', alpha=0.6)

# 水泵功率对比 (Pump Power)
axes[2,0].plot(res_ctrl['time'], res_ctrl['yPpum'], label='Pump (Controlled)', color='#66b3ff', linewidth=1.5)
axes[2,0].plot(res_base['time'], res_base['yPpum'], label='Pump (Baseline)', color='#66b3ff', linestyle='--', alpha=0.6)

axes[2,0].set_ylabel('Power [W]')
axes[2,0].set_xlabel('Time [Days]')
axes[2,0].legend(loc='upper right', fontsize='x-small', ncol=2) # 使用两列图例以节省空间
axes[2,0].set_title('Infrastructure Components: Controlled vs Baseline')

# --- 右列：热力学状态 ---
# [0,1] 温度控制
axes[0,1].plot(res_ctrl['time'], res_ctrl['t_dry_bul'], label='Outdoor Temp', color='orange', alpha=0.6)
axes[0,1].plot(res_ctrl['time'], res_ctrl['tcw_dry_set'], label='TCWDry Setpoint', color='darkblue', linestyle='--')
axes[0,1].set_ylabel('Temperature [°C]')
axes[0,1].legend(loc='upper right')
axes[0,1].set_title('Dynamic Setpoint Control')

# [1,1] 水路侧温度
axes[1,1].plot(res_ctrl['time'], np.array(res_ctrl['yTCDUSup'])-273.15, label='T_CDU_Supply')
axes[1,1].plot(res_ctrl['time'], np.array(res_ctrl['yTCDURet'])-273.15, label='T_CDU_Return')
axes[1,1].plot(res_ctrl['time'], np.array(res_ctrl['yTCWLeaTow'])-273.15, label='T_Tower_Leaving')
axes[1,1].set_ylabel('Temperature [°C]')
axes[1,1].legend(loc='upper right')
axes[1,1].set_title('Water Loop Temperatures')

# [2,1] TES 状态与控制信号
axes[2,1].plot(res_ctrl['time'], res_ctrl['ySOCtes'], color='green', label='SOC')
ax_tes = axes[2,1].twinx()
ax_tes.step(res_ctrl['time'], res_ctrl['sig_tes'], color='purple', alpha=0.3, label='SigTES', where='post')
axes[2,1].set_ylabel('SOC [-]')
ax_tes.set_ylabel('SigTES (1:CH, -1:DIS)')
axes[2,1].set_xlabel('Time [Days]')
axes[2,1].set_title('TES Status & Signal')

for ax in axes.flat:
    ax.grid(True, linestyle=':', alpha=0.6)

plt.show()