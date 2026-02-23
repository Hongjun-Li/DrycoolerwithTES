import numpy as np
import matplotlib.pyplot as plt
from pyfmi import load_fmu
from tqdm import tqdm

# 1. 加载 FMU
fmu_path = 'ASHRAE26_ChillerPlant_0tes_DataCenterDryFMU.fmu'
model = load_fmu(fmu_path)

# 仿真参数（第1天到第7天）
SIM_START_DAY = 1
SIM_END_DAY = 7
start_time = 86400.0 * (SIM_START_DAY - 1)
stop_time = 86400.0 * SIM_END_DAY
step_size = 60.0             # 1分钟步长
steps = int((stop_time - start_time) / step_size)

# 定义变量列表
plot_vars = ['yPHVAC', 'yPIT', 'yPDCTFan', 'yPpum', 'yTdry', 'yTCWLeaTow', 'yTCDUSup', 'yTCDURet', 'ySOCtes']

def get_electricity_price(time_seconds):
    """
    定义电价曲线 (单位: 元/kWh)
    00:00 - 08:00: 0.5 (谷)
    08:00 - 12:00: 1.2 (峰)
    12:00 - 17:00: 0.8 (平)
    17:00 - 21:00: 1.2 (峰)
    21:00 - 24:00: 0.5 (谷)
    """
    hour = (time_seconds % 86400) / 3600
    if (8 <= hour < 12) or (17 <= hour < 21):
        return 1.2
    elif 12 <= hour < 17:
        return 0.8
    else:
        return 0.5

def run_simulation(use_control=True):
    model.reset()
    model.setup_experiment(start_time=start_time)
    model.enter_initialization_mode()
    model.exit_initialization_mode()

    data = {var: [] for var in plot_vars}
    data.update({
        'time': [], 'sig_tes': [], 't_dry_bul': [], 
        'tcw_dry_set': [], 'pue': [], 'price': [], 'cum_cost': []
    })
    
    current_time = start_time
    sig_tes = 0.0
    total_cost = 0.0

    desc = "仿真(受控模式)" if use_control else "仿真(基准模式)"
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
            sig_tes = 0.0

        # --- C. Dry Cooler Setpoint 调节 ---
        t_base = min(43 + 273.15, max(35 + 273.15, t_db + 5.0))
        t_set = t_base - 3.0 if (sig_tes == 1.0 and is_night and use_control) else t_base

        # --- D. 执行步进 ---
        model.set('SigTES', sig_tes)
        model.set('TCWDry', t_set)
        model.do_step(current_time, step_size, True)

        # --- E. 成本与能效计算 ---
        c_pit = model.get('yPIT')[0]
        c_phvac = model.get('yPHVAC')[0]
        price = get_electricity_price(current_time)
        
        # 瞬时电量 (kWh) = 功率(W) * 时间(s) / 3,600,000
        # 仅考虑 HVAC 电费，不计 IT 电费
        energy_step_kwh = c_phvac * (step_size / 3600000.0)
        total_cost += energy_step_kwh * price
        
        data['time'].append(current_time / 86400)
        data['sig_tes'].append(sig_tes)
        data['t_dry_bul'].append(t_db - 273.15)
        data['tcw_dry_set'].append(t_set - 273.15)
        data['pue'].append((c_pit + c_phvac) / c_pit if c_pit > 0 else 1.0)
        data['price'].append(price)
        data['cum_cost'].append(total_cost)
        for var in plot_vars:
            data[var].append(model.get(var)[0])
        
        current_time += step_size
    return data

# 运行两次仿真对比
res_base = run_simulation(use_control=False)
res_ctrl = run_simulation(use_control=True)

# 计算节约金额
final_saving = res_base['cum_cost'][-1] - res_ctrl['cum_cost'][-1]
saving_percent = (final_saving / res_base['cum_cost'][-1]) * 100

print(f"\n--- HVAC电费经济效益分析 ---")
print(f"基准总HVAC电费 (Baseline): {res_base['cum_cost'][-1]:.2f} 元")
print(f"控制总HVAC电费 (Controlled): {res_ctrl['cum_cost'][-1]:.2f} 元")
print(f"HVAC节省总金额 (Total Saving): {final_saving:.2f} 元 ({saving_percent:.2f}%)")

# 4. 绘图：4行 x 2列 布局 (增加电价和成本)
fig, axes = plt.subplots(4, 2, figsize=(12, 7), sharex=True)
plt.subplots_adjust(hspace=0.35, wspace=0.25) 

# [0,0] PHVAC 对比
axes[0,0].plot(res_ctrl['time'], res_ctrl['yPHVAC'], label='Controlled', color='red')
axes[0,0].plot(res_base['time'], res_base['yPHVAC'], label='Baseline', color='blue', linestyle='--', alpha=0.6)
axes[0,0].set_ylabel('HVAC Power [W]')
axes[0,0].legend(loc='upper right', fontsize='small')
axes[0,0].set_title('Cooling Power Consumption')

# [1,0] PUE 对比
axes[1,0].plot(res_ctrl['time'], res_ctrl['pue'], color='darkred', label='Controlled PUE')
axes[1,0].set_ylim(1.0, 1.2)
axes[1,0].set_ylabel('PUE [-]')
axes[1,0].set_title('Power Usage Effectiveness')

# [2,0] 累计HVAC电费支出
axes[2,0].plot(res_ctrl['time'], res_ctrl['cum_cost'], label='Controlled Cost', color='green')
axes[2,0].plot(res_base['time'], res_base['cum_cost'], label='Baseline Cost', color='gray', linestyle='--')
axes[2,0].set_ylabel('Cum. HVAC Cost [Yuan]')
axes[2,0].legend(loc='upper left')
axes[2,0].set_title('Cumulative HVAC Electricity Cost')

# [3,0] 累计节约金额曲线
saving_curve = np.array(res_base['cum_cost']) - np.array(res_ctrl['cum_cost'])
axes[3,0].fill_between(res_ctrl['time'], saving_curve, color='gold', alpha=0.3, label='Total Saving')
axes[3,0].plot(res_ctrl['time'], saving_curve, color='orange', linewidth=2)
axes[3,0].set_ylabel('Savings [Yuan]')
axes[3,0].set_xlabel('Time [Days]')
axes[3,0].set_title(f'Total Price Saving: {final_saving:.2f} Yuan')

# [0,1] 温度与电价曲线 (右侧 Y 轴)
axes[0,1].plot(res_ctrl['time'], res_ctrl['t_dry_bul'], label='Outdoor Temp', color='orange', alpha=0.4)
ax_price = axes[0,1].twinx()
ax_price.step(res_ctrl['time'], res_ctrl['price'], where='post', color='black', label='Electricity Price')
ax_price.set_ylabel('Price [Yuan/kWh]')
axes[0,1].set_ylabel('Temp [°C]')
axes[0,1].set_title('Environment & Electricity Price')

# [1,1] 水路侧温度
axes[1,1].plot(res_ctrl['time'], np.array(res_ctrl['yTCDUSup'])-273.15, label='T_CDU_Sup')
axes[1,1].plot(res_ctrl['time'], np.array(res_ctrl['yTCDURet'])-273.15, label='T_CDU_Ret')
axes[1,1].set_ylabel('Temp [°C]')
axes[1,1].legend(loc='upper right', fontsize='x-small')
axes[1,1].set_title('Water Loop Temperatures')

# [2,1] 组件功率对比
axes[2,1].plot(res_ctrl['time'], res_ctrl['yPDCTFan'], label='Fan (Ctrl)', color='#ff9999')
axes[2,1].plot(res_base['time'], res_base['yPDCTFan'], label='Fan (Base)', color='#ff9999', linestyle='--', alpha=0.5)
axes[2,1].set_ylabel('Power [W]')
axes[2,1].legend(loc='upper right', fontsize='x-small')
axes[2,1].set_title('Component Power Analysis')

# [3,1] TES 状态
axes[3,1].plot(res_ctrl['time'], res_ctrl['ySOCtes'], color='green', label='SOC')
ax_sig = axes[3,1].twinx()
ax_sig.step(res_ctrl['time'], res_ctrl['sig_tes'], color='purple', alpha=0.3, where='post')
axes[3,1].set_ylabel('SOC [-]')
ax_sig.set_ylabel('SigTES')
axes[3,1].set_xlabel('Time [Days]')
axes[3,1].set_title('TES Battery Operation')

for ax in axes.flat: ax.grid(True, linestyle=':', alpha=0.6)
plt.show()