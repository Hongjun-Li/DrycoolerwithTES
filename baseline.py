import numpy as np
import matplotlib.pyplot as plt
from pyfmi import load_fmu
from pathlib import Path

# 1. 加载 FMU 模型
# 请将 'datacenter_cooling.fmu' 替换为你的实际文件路径
model = load_fmu('ASHRAE26_ChillerPlant_0tes_DataCenterDryFMU.fmu')

# 2. 仿真参数设置
start_time = 0.0
days = 7
end_time = 86400.0 * days  
step_size = 900.0          # 控制步长：15分钟 (900秒)
current_time = start_time

# ASHRAE 天气文件：读取 weather/3C.mos 的干球温度（C2）
MOS_PATH = Path('weather') / '3C.mos'


def load_mos_tdry_profile(mos_path: Path):
    if not mos_path.exists():
        raise FileNotFoundError(f"MOS 文件不存在: {mos_path}")

    time_s = []
    drybulb_c = []
    with mos_path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            stripped = line.strip()
            if (not stripped) or stripped.startswith('#'):
                continue

            parts = stripped.split()
            if len(parts) < 2:
                continue

            try:
                t_s = float(parts[0])
                t_c = float(parts[1])
            except ValueError:
                continue

            time_s.append(t_s)
            drybulb_c.append(t_c)

    if len(time_s) < 2:
        raise ValueError("MOS 文件中可用天气数据不足（需要至少2行数值数据）")

    return np.array(time_s, dtype=float), np.array(drybulb_c, dtype=float)


mos_time_s, mos_tdry_c = load_mos_tdry_profile(MOS_PATH)


def mos_tdry_k(sim_time_s):
    t_c = np.interp(sim_time_s, mos_time_s, mos_tdry_c)
    return t_c + 273.15


# yTES 控制状态机参数（按图中阈值）
SOC_CHARGE_ON = 0.50
SOC_DISCHARGE_ON = 0.65
SOC_CHARGE_OFF = 0.70
SOC_DISCHARGE_OFF = 0.05
DAY_START_HOUR = 6.0
DAY_END_HOUR = 22.0

# 图中含有干球温度使能条件（TDryBul > 273.15 + ...），此处给出可调默认值
T_DRY_ENABLE_K = 273.15 + 20.0

# TES 状态：off / charge / discharge
tes_mode = 'off'

# 初始化模型
model.initialize(start_time, end_time)

# 用于存储数据的列表
data = {
    'time': [],
    'soc': [],
    't_drybulb': [],
    't_cw_lea_tow': [],
    't_setpoint': [],
    'tes_signal': []
}

# 3. 仿真与控制循环
while current_time < end_time:
    # --- A. 获取当前状态 (读取 FMU 输出) ---
    # 将 weather/3C.mos 的干球温度写入 FMU 输入 yTdry
    t_drybulb = mos_tdry_k(current_time)
    model.set('yTdry', t_drybulb)

    # 读取 SOC 和其他输出
    soc = model.get('ySOCtes')[0]
    t_cw_lea_tow = model.get('yTCWLeaTow')[0]
    
    # --- B. 运行控制逻辑 (RBC) ---
    # 判断昼夜
    hour_of_day = (current_time % 86400) / 3600
    is_day = (DAY_START_HOUR <= hour_of_day < DAY_END_HOUR)
    is_night = not is_day

    # 1. 计算 Drycooler Setpoint (TCWDry)
    # 基础参考值
    t_base = min(43 + 273.15, max(273.15 + 35, t_drybulb + 5))
    if is_night:
        t_setpoint = t_base - 3.0  # 夜间降低 3 度
    else:
        t_setpoint = t_base
        
    # 2. 计算 TES 控制信号 (yTES: -1 到 1)
    # 按图逻辑：
    # - off -> charge: 夜间且 SOC < 0.5
    # - off -> discharge: 白天且 SOC > 0.65
    # - charge -> off: SOC >= 0.7
    # - discharge -> off: SOC <= 0.05
    # - 干球温度使能：TDryBul > 273.15 + 20°C
    dry_enable = (t_drybulb > T_DRY_ENABLE_K)

    if tes_mode == 'off':
        if dry_enable and is_day and soc > SOC_DISCHARGE_ON:
            tes_mode = 'discharge'
        elif dry_enable and is_night and soc < SOC_CHARGE_ON:
            tes_mode = 'charge'
    elif tes_mode == 'charge':
        if (soc >= SOC_CHARGE_OFF) or (not dry_enable):
            tes_mode = 'off'
    elif tes_mode == 'discharge':
        if (soc <= SOC_DISCHARGE_OFF) or (not dry_enable):
            tes_mode = 'off'

    if tes_mode == 'charge':
        tes_signal = 1.0
    elif tes_mode == 'discharge':
        tes_signal = -1.0
    else:
        tes_signal = 0.0

    # --- C. 将控制信号写入模型 ---
    model.set('TCWDry', t_setpoint)
    model.set('yTES', tes_signal)
    
    # --- D. 记录数据用于绘图 ---
    data['time'].append(current_time / 3600)  # 转换为小时方便看图
    data['soc'].append(soc)
    data['t_drybulb'].append(t_drybulb - 273.15) # 转换为摄氏度
    data['t_cw_lea_tow'].append(t_cw_lea_tow - 273.15) # 转换为摄氏度
    data['t_setpoint'].append(t_setpoint - 273.15) # 转换为摄氏度
    data['tes_signal'].append(tes_signal)
    
    # --- E. 步进仿真 ---
    model.do_step(current_time, step_size, True)
    current_time += step_size

# 4. 结果可视化
t_dry_min = min(data['t_drybulb'])
t_dry_max = max(data['t_drybulb'])
t_dry_span = t_dry_max - t_dry_min
print(f"yTdry range: min={t_dry_min:.3f}°C, max={t_dry_max:.3f}°C, span={t_dry_span:.3f}°C")
if t_dry_span < 0.05:
    print("Warning: yTdry 几乎恒定。通常是 FMU 天气边界未随时间变化（如固定干球温度参数）导致。")

plt.figure(figsize=(8, 6))

# 图 1: 室外干球温度（单独一张）
plt.subplot(3, 1, 1)
plt.plot(data['time'], data['t_drybulb'], label='T_drybulb (Outdoor)', color='orange')
plt.ylabel('Temperature (°C)')
plt.title('Outdoor Dry Bulb Temperature')
plt.grid(True)
plt.legend()

# 图 2: 冷却塔出水温度（yTCWLeaTow）
plt.subplot(3, 1, 2)
plt.plot(data['time'], data['t_cw_lea_tow'], label='TCW Leaving Tower Temperature', color='teal')
plt.ylabel('Temperature (°C)')
plt.title('yTCWLeaTow')
plt.grid(True)
plt.legend()

# 图 2: SOC 和 TES 控制信号（同一张图，双y轴）
ax1 = plt.subplot(3, 1, 3)
line1, = ax1.plot(data['time'], data['soc'], label='TES SOC', color='green')
ax1.set_ylabel('State of Charge (0-1)', color='green')
ax1.set_ylim(-0.1, 1.1)
ax1.tick_params(axis='y', labelcolor='green')
ax1.grid(True)

ax2 = ax1.twinx()
line2, = ax2.step(data['time'], data['tes_signal'], where='post', label='yTES (-1=Discharge, 1=Charge)', color='purple')
ax2.set_ylabel('TES Control Signal', color='purple')
ax2.set_yticks([-1, 0, 1])
ax2.set_ylim(-1.2, 1.2)
ax2.tick_params(axis='y', labelcolor='purple')

ax1.set_xlabel('Time (Hours)')
ax1.legend([line1, line2], [line1.get_label(), line2.get_label()], loc='upper right')

plt.tight_layout()
plt.show()