import numpy as np
import matplotlib.pyplot as plt
from pyfmi import load_fmu
from tqdm import tqdm

# 1. 加载 FMU 并准备变量
fmu_path = 'ASHRAE26_ChillerPlant_0tes_DataCenterDryFMU.fmu'
model = load_fmu(fmu_path)

# 仿真参数
start_time = 0.0
stop_time = 3600.0 * 24 * 7  # 7天
step_size = 60.0             # 1分钟步长

# 定义输出变量列表 (根据最新模型图)
plot_vars = [
    'yPHVAC', 'yPIT', 'yPDCTFan', 'yPpum', 
    'yTdry', 'yTCWLeaTow', 'yTCDUSup', 'yTCDURet', 'ySOCtes'
]

# 初始化存储字典
res = {var: [] for var in plot_vars}
res.update({'time': [], 'sig_tes': [], 't_dry_bul': [], 'tcw_dry_set': [], 'pue': []})

# 2. 初始化模型状态
model.setup_experiment(start_time=start_time)
model.enter_initialization_mode()
model.exit_initialization_mode()

current_time = start_time
sig_tes = 0.0

# 3. 仿真循环 + tqdm 进度条
print(f"正在执行 2列布局 闭环仿真...")
for _ in tqdm(range(int(stop_time / step_size)), desc="仿真进度"):
    # --- A. 获取传感器读数 ---
    soc = model.get('ySOCtes')[0]
    try:
        t_db = model.get('weaBus.TDryBul')[0]
    except:
        t_db = 293.15

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

    # --- C. Dry Cooler Setpoint 调节 (根据逻辑图 image_3b4ccb.png) ---
    t_base = min(43 + 273.15, max(35 + 273.15, t_db + 5.0))
    # 夜间充电降温补偿
    t_set = t_base - 3.0 if (sig_tes == 1.0 and is_night) else t_base

    # --- D. 执行步进 ---
    model.set('SigTES', sig_tes)
    model.set('TCWDry', t_set)
    model.do_step(current_time, step_size, True)

    # --- 仿真循环内部的 PUE 计算修正 ---
    current_pit = model.get('yPIT')[0]
    current_phvac = model.get('yPHVAC')[0]

    # 修正公式：PUE = (IT功率 + 冷却功率) / IT功率
    pue_value = (current_pit + current_phvac) / current_pit if current_pit > 0 else 1.0
    
    
    res['time'].append(current_time / 86400)
    res['sig_tes'].append(sig_tes)
    res['t_dry_bul'].append(t_db - 273.15)
    res['tcw_dry_set'].append(t_set - 273.15)
    res['pue'].append(pue_value)
    for var in plot_vars:
        res[var].append(model.get(var)[0])
    
    current_time += step_size

# 4. 绘图：2列布局 (3行 x 2列)
fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)

# 修正：将 wwidth 改为 wspace
plt.subplots_adjust(hspace=0.3, wspace=0.25) 

# --- 左列：电力与效能 ---
# (0,0) 总功耗 vs IT功耗
axes[0,0].plot(res['time'], res['yPHVAC'], label='Total PHVAC', color='black', linewidth=1.5)
axes[0,0].fill_between(res['time'], res['yPIT'], color='skyblue', alpha=0.5, label='IT Power (yPIT)')
axes[0,0].set_ylabel('Power [W]')
axes[0,0].legend(loc='upper right')
axes[0,0].set_title('Power Consumption: Total vs IT')

# (1,0) PUE 计算值
avg_pue = np.mean(res['pue'])
axes[1,0].plot(res['time'], res['pue'], color='red', label='Real-time PUE')
axes[1,0].axhline(y=avg_pue, color='darkred', linestyle='--', label=f'Avg PUE: {avg_pue:.3f}')

axes[1,0].set_ylim(0.75, 1.5) 
axes[1,0].set_ylabel('PUE [-]')
axes[1,0].legend(loc='upper right')
axes[1,0].set_title('Power Usage Effectiveness (PUE)')


# (2,0) 辅机功耗：水泵与风机
# 使用 stackplot 堆叠显示风机和水泵功率
axes[2,0].stackplot(res['time'], 
                    res['yPDCTFan'], res['yPpum'], 
                    labels=['Fan (yPDCTFan)', 'Pump (yPpum)'], 
                    colors=['#ff9999', '#66b3ff'], alpha=0.8)

# 同时绘制 yPHVAC 实线作为对照（如果 PHVAC 包含这两者以外的损耗，线会略高于堆叠区域）
axes[2,0].plot(res['time'], res['yPHVAC'], color='black', label='Total Cooling (yPHVAC)', linewidth=1.2)

axes[2,0].set_ylabel('Power [W]')
axes[2,0].set_xlabel('Time [Days]')
axes[2,0].legend(loc='upper right')
axes[2,0].set_title('Cooling Power Breakdown: Fan + Pump vs Total')

# --- 右列：热力学状态 ---
# (0,1) 温度控制：室外 vs 设定值
axes[0,1].plot(res['time'], res['t_dry_bul'], label='Outdoor Temp', color='orange', alpha=0.6)
axes[0,1].plot(res['time'], res['tcw_dry_set'], label='TCWDry Setpoint', color='darkblue', linestyle='--')
axes[0,1].set_ylabel('Temperature [°C]')
axes[0,1].legend(loc='upper right')
axes[0,1].set_title('Dynamic Setpoint Control')

# (1,1) 水路侧关键温度 (从 K 转为 °C)
axes[1,1].plot(res['time'], np.array(res['yTCDUSup'])-273.15, label='T_CDU_Supply')
axes[1,1].plot(res['time'], np.array(res['yTCDURet'])-273.15, label='T_CDU_Return')
axes[1,1].plot(res['time'], np.array(res['yTCWLeaTow'])-273.15, label='T_Tower_Leaving')
axes[1,1].set_ylabel('Temperature [°C]')
axes[1,1].legend(loc='upper right')
axes[1,1].set_title('Water Loop Temperatures')

# (2,1) TES 状态与控制信号
axes[2,1].plot(res['time'], res['ySOCtes'], color='green', label='SOC')
ax_tes = axes[2,1].twinx()
ax_tes.step(res['time'], res['sig_tes'], color='purple', alpha=0.3, label='SigTES', where='post')
axes[2,1].set_ylabel('SOC [-]')
ax_tes.set_ylabel('SigTES (1:CH, -1:DIS)')
axes[2,1].set_xlabel('Time [Days]')
axes[2,1].set_title('TES Status & Signal')

for ax in axes.flat:
    ax.grid(True, linestyle=':', alpha=0.6)

plt.show()