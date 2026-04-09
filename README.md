# 🧠 Animal Behaviour Anomaly Detection using AI

🚀 An intelligent system that detects anomalies in animal behaviour using deep learning and visualizes results in real-time through an interactive dashboard.

---

## 📌 Project Overview

Animals often exhibit unusual behaviour before environmental disturbances such as earthquakes, storms, or habitat changes. This project leverages **Artificial Intelligence (LSTM Autoencoder)** to analyze behavioural data and detect anomalies.

The system converts complex anomaly scores into a **Stability Index (0–100)** and presents it through a **real-time dashboard**, making it easy to monitor and interpret behavioural patterns.

---

## 🎯 Key Features

✅ AI-based anomaly detection using LSTM
✅ Stability Index generation (0–100 scale)
✅ Real-time dashboard (Streamlit)
✅ Stock market–like dynamic graph 📈
✅ Scenario-based simulation (Normal → Severe)
✅ Weather integration 🌦️
✅ Live monitoring table
✅ Cloud deployment using Cloudflare Tunnel

---

## 🏗️ System Architecture

```text
Dataset → Preprocessing → LSTM Model → Anomaly Detection → Stability Index → Dashboard Visualization
```

---

## 🧠 Machine Learning Model

* **Model Used:** LSTM Autoencoder
* **Type:** Unsupervised Learning
* **Purpose:** Learn normal behaviour and detect deviations

📌 **Working Principle:**

* Input sequence → Model reconstruction
* Calculate error (MSE)
* Higher error = anomaly

---

## 🖥️ Tech Stack

| Category      | Technology Used    |
| ------------- | ------------------ |
| Language      | Python             |
| ML Framework  | TensorFlow / Keras |
| Data Handling | Pandas, NumPy      |
| Visualization | Streamlit, Plotly  |
| Deployment    | Cloudflare Tunnel  |
| API           | Weather API        |

---

## 📸 Dashboard Preview

View the uploaded video 

---

## ⚙️ Installation & S

### 1 Install Dependencies

```bash
pip install -r requirements.txt
```

### 2 Run Backend (Model)

```bash
python rp.py
```

### 3 Run Dashboard

```bash
streamlit run DASH.py
```

### 4 Open in Browser

```
http://localhost:8501 (The link will get after running the final command this is an just demo )
```

---

## 📊 Output

* Stability Index (0–100)
* Behaviour Status:

  * 🟢 Normal
  * 🟡 Warning
  * 🔴 Critical
* Live Graph Visualization
* Real-time Monitoring Table

---

## 🔬 Real-World Applications

🌍 Disaster prediction
🐾 Wildlife monitoring
🌱 Environmental analysis
🚨 Early warning systems

---

## ⚠️ Limitations

* Uses simulated dataset
* No real-time sensor integration
* Accuracy depends on data quality

---

## 🚀 Future Enhancements

* IoT sensor integration
* Mobile application
* Cloud deployment (AWS/GCP)
* Advanced AI models (Transformers)
* Multi-species behaviour analysis

---


## 👨‍💻 Author

**Prajwal Adhav**
MSCIT  Project

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!

---
