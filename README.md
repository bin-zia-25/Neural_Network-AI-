# 🛡️ AI-Powered System Health Monitor

A real-time system stability predictor built with **Deep Learning** and optimized using the **Singleton Design Pattern**. This application monitors hardware metrics (CPU, RAM, Disk I/O) and predicts the likelihood of system instability.

## 🚀 Live Demo
Link: https://zcmq9cuk2hjz6oceqjiejh.streamlit.app/

## 🧠 Technical Architecture

### 1. Singleton Design Pattern
To ensure memory efficiency, the Deep Learning model is wrapped in a **Singleton Class**. This prevents the application from reloading the heavy `.h5` model file multiple times, saving significant RAM and improving prediction speed.

### 2. Deep Learning Model
The "brain" of the project is a **Sequential Neural Network** built with TensorFlow/Keras:
- **Input Layer:** 3 Neurons (CPU%, RAM%, Disk%)
- **Hidden Layers:** Dense layers with **ReLU** activation to identify complex resource bottleneck patterns.
- **Output Layer:** A single neuron with a **Sigmoid** activation function, providing a probability score (0.0 to 1.0) of system instability.



[Image of a neural network with input hidden and output layers]


## 🛠️ Tech Stack
- **Language:** Python
- **Framework:** Streamlit (Web UI)
- **AI/ML:** TensorFlow, Keras, NumPy
- **System Metrics:** Psutil
- **Architecture:** Singleton Design Pattern

## 📊 Features
- **Real-time Monitoring:** Fetches live kernel data using `psutil`.
- **Visualized Thinking:** Includes a Graphviz-generated diagram of the neural network layers.
- **Predictive Alerts:** Dynamically switches between Healthy (Success), Warning, and Critical (Error) states based on AI confidence.
- **Cross-Platform:** Designed to run on both Windows (Local) and Linux (Cloud).
