# DASH.py 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
import random
from datetime import datetime
import pytz
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI Behaviour Stability Monitor", layout="wide")

st.title("📡 AI-Based Animal Behaviour Stability Monitoring")

# ----------------------------------------------------
# WEATHER API
# ----------------------------------------------------

API_KEY = " PASTE YOUR WEATHER API KEY"
CITY = "Navi Mumbai"

def get_weather():
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
        data = requests.get(url, timeout=5).json()

        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        condition = data["weather"][0]["main"]

        return temp, humidity, condition

    except:
        return 30, 60, "Clear"

temp, humidity, condition = get_weather()

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("stability_output.csv")

df = load_data()

if df.empty:
    st.error("No stability data found")
    st.stop()

base_stability = df["stability"].values.copy()

# ----------------------------------------------------
# SESSION STATE EVENT LOG
# ----------------------------------------------------

if "event_log" not in st.session_state:
    st.session_state.event_log = pd.DataFrame(
        columns=["Time","Stability","Status","Event"]
    )

# ----------------------------------------------------
# LIVE DATA WINDOW
# ----------------------------------------------------

WINDOW = 120
live_data = base_stability[-WINDOW:].copy()

# ----------------------------------------------------
# STREAMLIT PLACEHOLDERS
# ----------------------------------------------------

status_box = st.empty()
weather_box = st.empty()
event_box = st.empty()
graph_box = st.empty()
table_box = st.empty()

# ----------------------------------------------------
# IST TIMEZONE
# ----------------------------------------------------

ist = pytz.timezone("Asia/Kolkata")

# ----------------------------------------------------
# LIVE LOOP
# ----------------------------------------------------

for _ in range(1000000):

    live_data = np.roll(live_data, -1)

    noise = random.uniform(-0.4,0.4)
    new_value = live_data[-1] + noise
    new_value = max(60, min(100, new_value))

    live_data[-1] = new_value

    avg = np.mean(live_data)

    # ------------------------------------------------
    # AI PREDICTION
    # ------------------------------------------------

    X = np.arange(len(live_data)).reshape(-1,1)
    y = live_data

    model = LinearRegression()
    model.fit(X,y)

    future = 10
    future_x = np.arange(len(live_data), len(live_data)+future).reshape(-1,1)

    forecast = model.predict(future_x)

    # ------------------------------------------------
    # STATUS
    # ------------------------------------------------

    if avg > 90:
        status = "🟢 NORMAL"
    elif avg > 80:
        status = "🟡 WARNING"
    else:
        status = "🔴 CRITICAL"

    # ------------------------------------------------
    # EVENT DETECTOR
    # ------------------------------------------------

    event = "Stable"

    if live_data[-1] < 75:
        event = "⚠ Behaviour Instability"

    elif temp > 35:
        event = "⚠ Heat Stress Impact"

    elif humidity > 85:
        event = "⚠ Climate Disturbance"

    # ------------------------------------------------
    # STATUS PANEL
    # ------------------------------------------------

    status_box.markdown(f"""
### System Status: {status}
""")

    # ------------------------------------------------
    # WEATHER PANEL
    # ------------------------------------------------

    weather_box.markdown(f"""
### 🌦 Live Weather Data

Temperature: **{temp}°C**  
Humidity: **{humidity}%**  
Condition: **{condition}**

Last Updated: {datetime.now(ist).strftime('%H:%M:%S')}
""")

    # ------------------------------------------------
    # EVENT PANEL
    # ------------------------------------------------

    event_box.markdown(f"""
### 🤖 AI Behaviour Event Detector

{event}
""")

    # ------------------------------------------------
    # GRAPH
    # ------------------------------------------------

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=live_data,
        mode="lines",
        name="Live Stability",
        line=dict(width=3)
    ))

    fig.add_trace(go.Scatter(
        x=np.arange(len(live_data), len(live_data)+future),
        y=forecast,
        mode="lines",
        name="AI Forecast",
        line=dict(dash="dash")
    ))

    fig.update_layout(
        template="plotly_dark",
        height=450,
        title="📈 Live Behaviour Stability Stream"
    )

    graph_box.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------
    # EVENT TABLE
    # ------------------------------------------------

    current_time = datetime.now(ist).strftime("%H:%M:%S")

    new_row = pd.DataFrame({
        "Time":[current_time],
        "Stability":[round(live_data[-1],2)],
        "Status":[status],
        "Event":[event]
    })

    # insert row at top
    st.session_state.event_log = pd.concat(
        [new_row, st.session_state.event_log],
        ignore_index=True
    )

    st.session_state.event_log = st.session_state.event_log.head(10)

    table_box.dataframe(
        st.session_state.event_log,
        use_container_width=True
    )

    time.sleep(2)
    
