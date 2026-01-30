import streamlit as st
import psutil
import numpy as np
import platform
import time
import pandas as pd
from keras.models import load_model
import graphviz

# --- THE SINGLETON CLASS ---
class ModelSingleton:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelSingleton, cls).__new__(cls)
            cls._instance.model = load_model('system_health_model.h5')
            cls._instance.load_time = time.strftime("%H:%M:%S")
        return cls._instance

# Initialize
brain = ModelSingleton()

# --- SIDEBAR: SYSTEM INFO & SINGLETON STATUS ---
st.sidebar.title("⚙️ System Internals")
st.sidebar.info(f"**Singleton Status:** Active\n\n**Model Loaded At:** {brain.load_time}")
st.sidebar.write("Using a single instance of the model ensures Windows RAM is protected.")

st.sidebar.subheader("📍 Deployment Info")
st.sidebar.write(f"**Processor:** {platform.processor()}")
st.sidebar.write(f"**OS:** {platform.system()}") # This will say 'Linux' on the web!

# --- MAIN UI ---
st.title("🛡️ AI System Health Monitor")

# 1. NEURAL NETWORK VISUALIZATION
with st.expander("🧠 View Neural Network Architecture", expanded=False):
    dot = graphviz.Digraph(comment='Model Architecture')
    dot.attr(rankdir='LR', size='8,5')
    
    # Define Layers
    inputs = ['CPU', 'RAM', 'Disk']
    hiddens = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8']
    output = ['Stability']
    
    # Add Nodes
    for i in inputs: dot.node(i, color='lightblue2', style='filled')
    for h in hiddens: dot.node(h, color='lightgrey', style='filled')
    dot.node(output[0], color='lightgreen', style='filled')
    
    # Connect Layers (Simplified for UI)
    for i in inputs:
        for h in hiddens: dot.edge(i, h)
    for h in hiddens: dot.edge(h, output[0])
    
    st.graphviz_chart(dot)
    st.caption("Simplified representation of your Deep Learning layers (3 -> 8 -> 4 -> 1)")

# --- LIVE DASHBOARD ---
metrics_area = st.empty()
chart_area = st.empty()

# History storage
if 'history' not in st.session_state:
    st.session_state.history = []

while True:
    # 1. DATA COLLECTION
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory().percent                           
    disk = psutil.disk_usage('/').percent

    # 2. PREDICTION (with safety clamp)
    input_data = np.array([[cpu/100, ram/100, disk/100]])
    prediction_raw = brain.model.predict(input_data)
    val = float(prediction_raw[0][0])
    prediction = max(0.0, min(1.0, val))

    # Save to history for the chart
    st.session_state.history.append(prediction)
    if len(st.session_state.history) > 20: # Keep only last 20 readings
        st.session_state.history.pop(0)

    # 3. DISPLAY RESULTS
    with metrics_area.container():
        col1, col2, col3 = st.columns(3)
        col1.metric("CPU", f"{cpu}%")
        col2.metric("RAM", f"{ram}%")
        col3.metric("Disk", f"{disk}%")
        
        st.write(f"### AI Stability Confidence: {round(prediction * 100, 2)}%")
        st.progress(prediction)

        if prediction > 0.8:
        # High Risk: Red Alert
            st.error(f"🚨 CRITICAL STATE: System instability is at {round(prediction*100)}%!")
            st.warning("Action Suggested: Close heavy background processes or check Task Manager.")

        elif prediction > 0.5:
        # Moderate Risk: Yellow Warning
            st.warning(f"⚠️ CAUTION: System load is increasing ({round(prediction*100)}%).")
            st.info("The AI detects patterns similar to a system slowdown.")
        else:
            # Healthy: Green Message
            st.success("✅ SYSTEM STABLE: No instability patterns detected.")

    # 4. TREND CHART
    with chart_area.container():
        st.write("### Prediction Trend (Last 20 Samples)")
        st.line_chart(st.session_state.history)

    time.sleep(1)
