import numpy as np
from pyfmi import load_fmu

# 1. 加载 FMU 文件
fmu_path = 'ASHRAE26_ChillerPlant_0tes_DataCenterDryFMU.fmu'  # 请替换为你实际的 FMU 文件路径
model = load_fmu(fmu_path)

# 2. 定义仿真参数
start_time = 0.0
stop_time = 3600.0*24*7  # 仿真 1 小时（根据需要调整）
setpoint_value = 35 + 273.15  # 308.15 K

# 3. 设置输入信号
# 在 pyfmi 中，输入信号通常需要是一个包含时间戳和数值的结构
# 假设 FMU 的输入变量名为 'TCWDry' (根据图片推断)
# 如果不确定变量名，可以使用 model.get_model_variables().keys() 查看
input_name = 'TCWDry' 
input_data = np.transpose(np.vstack(([start_time, stop_time], 
                                     [setpoint_value, setpoint_value])))

input_object = (input_name, input_data)

# 4. 执行仿真
# opts = model.simulate_options() # 可以根据需要配置 solver
res = model.simulate(start_time=start_time, 
                     final_time=stop_time, 
                     input=input_object)

# 5. 获取输出结果
# 根据图片，输出变量名应该是 'yPHVAC'
phvac_results = res['yPHVAC']
time_axis = res['time']

# 6. 打印或处理结果
print(f"仿真完成。")
print(f"最终 PHVAC 输出值: {phvac_results[-1]}")

# 如果需要简单的可视化
import matplotlib.pyplot as plt
plt.plot(time_axis, phvac_results)
plt.xlabel('Time [s]')
plt.ylabel('PHVAC')
plt.title('Simulation Results')
plt.grid(True)
plt.show()