import streamlit as st
import numpy as np
import cvxpy as cp
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 0. 隐私与保密提示 ---
# 您放心，此代码在您的本地环境运行，不会上传任何数据。
st.set_page_config(page_title="Chiller-less MPC 模拟", layout="wide")

st.title("❄️ 无冷机 (Chiller-less) 数据中心 TES 优化控制")
st.markdown("""
**场景设定：**
* **固定 IT 负载**：服务器功耗恒定。
* **限制条件**：没有冷机，仅靠 **Dry Cooler (干冷器)**。
* **物理挑战**：Dry Cooler 的制冷能力和效率严重依赖 **外界干球温度 (Dry Bulb Temp)**。
* **优化目标**：在气温高或电价高时，利用 **TES (蓄冷罐)** 放电，实现全天候平稳运行。
""")

# --- 1. 参数设置 ---
with st.sidebar:
    st.header("⚙️ 环境与系统参数")
    T_horizon = 24  # 24小时预测
    
    # IT 负载设定
    it_load_kw = st.number_input("IT 固定热负载 (kW)", value=100.0)
    
    # TES 设定
    storage_capacity = st.number_input("TES 蓄冷罐容量 (kWh)", value=800.0)
    initial_storage = st.number_input("初始蓄冷量 (kWh)", value=200.0)
    
    # 波动性设定
    st.subheader("环境波动")
    base_temp = st.slider("夜间最低气温 (°C)", 5, 20, 10)
    temp_swing = st.slider("昼夜温差 (°C)", 5, 20, 15)
    
    st.subheader("设备性能")
    max_dry_cooler_cap = st.number_input("Dry Cooler 最大制冷能力 (kW)", value=300.0)

# --- 2. 模拟预测数据 (Forecast Generation) ---
def generate_scenario(steps):
    time = np.arange(steps)
    
    # 1. 恒定负载
    load = np.full(steps, it_load_kw)
    
    # 2. 干球温度 (Dry Bulb)：通常下午 2-3 点最热
    # 使用 sin 函数模拟昼夜温差
    temp = base_temp + temp_swing * (0.5 + 0.5 * np.sin(2 * np.pi * (time - 9) / 24))
    
    # 3. 电价：通常与气温正相关 (白天贵，晚上便宜)
    price = 0.10 + 0.10 * (0.5 + 0.5 * np.sin(2 * np.pi * (time - 8) / 24))
    
    # 4. 动态 COP (关键物理模型)
    # 假设：气温越低，Dry Cooler 效率越高。
    # 简单模型：COP = 20 - 0.5 * Temp。 如果气温超过35度，COP极低。
    cop = 20.0 - 0.6 * temp
    cop = np.maximum(cop, 2.0) # 设定一个最低 COP (风扇全速转)
    
    return time, load, temp, price, cop

time_steps, load_forecast, temp_forecast, price_forecast, cop_forecast = generate_scenario(T_horizon)

# --- 3. MPC 优化建模 (CVXPY) ---
# 变量
cooling_produced = cp.Variable(T_horizon) # Dry Cooler 产生的冷量 (kW)
stored_energy = cp.Variable(T_horizon + 1) # 蓄冷罐状态 (kWh)

# 目标：最小化电费
# 电费 = (产生的冷量 / COP) * 电价
# 注意：这是一个近似，为了保持凸优化 (Convex)，我们预先计算 "单位冷量的成本"
cost_per_unit_cooling = price_forecast / cop_forecast
total_cost = cp.sum(cp.multiply(cooling_produced, cost_per_unit_cooling))

constraints = [stored_energy[0] == initial_storage]

for t in range(T_horizon):
    # 能量守恒：下时刻蓄冷 = 当前蓄冷 + 产冷 - 负载消耗
    constraints += [stored_energy[t+1] == stored_energy[t] + cooling_produced[t] - load_forecast[t]]
    
    # 物理约束
    constraints += [cooling_produced[t] >= 0] # 不能产生负冷量
    # Dry Cooler 的能力受气温限制 (气温越高，最大制冷能力越低 - 模拟物理极限)
    # 假设气温每升高1度，最大能力下降 2% (仅作演示)
    current_max_cap = max_dry_cooler_cap * (1 - 0.02 * (temp_forecast[t] - 10)) 
    constraints += [cooling_produced[t] <= current_max_cap]
    
    constraints += [stored_energy[t+1] >= 0] # 罐子不能空
    constraints += [stored_energy[t+1] <= storage_capacity] # 罐子不能溢出

# 求解
problem = cp.Problem(cp.Minimize(total_cost), constraints)
problem.solve()

# --- 4. 结果展示 ---
if problem.status == "optimal":
    
    # 准备数据
    power_consumed = cooling_produced.value / cop_forecast
    tank_action = cooling_produced.value - load_forecast # 正=充电，负=放电
    
    st.success(f"✅ 优化完成！策略主要利用了夜间低温时段进行预冷。")
    
    # --- 可视化：关键图表 ---
    
    # 创建双轴图表：气温 vs 蓄冷量
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 1. 区域背景：干球温度 (这是我们的"敌人")
    fig.add_trace(
        go.Scatter(x=time_steps, y=temp_forecast, name="外界干球温度 (°C)",
                   line=dict(color='orange', width=2, dash='dot')),
        secondary_y=False
    )
    
    # 2. 蓄冷罐状态 (这是我们的"武器")
    fig.add_trace(
        go.Scatter(x=time_steps, y=stored_energy.value[:-1], name="TES 蓄冷量 (kWh)",
                   fill='tozeroy', line=dict(color='blue')),
        secondary_y=True
    )
    
    fig.update_layout(
        title_text="策略核心：气温升高前，蓄冷罐是否已充满？",
        hovermode="x unified"
    )
    fig.update_yaxes(title_text="干球温度 (°C)", secondary_y=False)
    fig.update_yaxes(title_text="蓄冷量 (kWh)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # --- 详细动作分析 ---
    st.subheader("🔍 详细动作分析")
    col1, col2 = st.columns(2)
    
    with col1:
        # 负载来源图
        fig2 = go.Figure()
        # 什么时候靠机器冷？
        fig2.add_trace(go.Bar(x=time_steps, y=cooling_produced.value, name="Dry Cooler 产冷", marker_color='lightblue'))
        # 什么时候靠罐子冷？(当产冷 < 负载)
        discharge = np.maximum(0, load_forecast - cooling_produced.value)
        fig2.add_trace(go.Bar(x=time_steps, y=discharge, name="TES 放冷 (补充)", marker_color='darkblue'))
        
        # IT Load 线
        fig2.add_trace(go.Scatter(x=time_steps, y=load_forecast, name="固定 IT 负载", line=dict(color='red', width=3)))
        
        fig2.update_layout(title="冷量来源分解", barmode='stack', yaxis_title="功率 (kW)")
        st.plotly_chart(fig2, use_container_width=True)
        
    with col2:
        # 效率 COP 图
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=time_steps, y=cop_forecast, name="系统 COP (随气温变化)", line=dict(color='green')))
        fig3.update_layout(title="系统能效 (COP) 变化", yaxis_title="COP")
        st.plotly_chart(fig3, use_container_width=True)

else:
    st.error("无法找到最优解。可能是TES容量太小，无法撑过高温时段，或者Dry Cooler最大功率不足。")