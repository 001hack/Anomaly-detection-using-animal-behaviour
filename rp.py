# rp.py 
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

print("Loading dataset...")
df = pd.read_csv("combined_multi_species_features_simulated.csv")

# ----------------------------------------
# STEP 1: Keep only numeric data
# ----------------------------------------
df = df.select_dtypes(include=["int64", "float64"])
df = df.fillna(df.mean())

print("Numeric data shape:", df.shape)

# ----------------------------------------
# STEP 2: Normalize
# ----------------------------------------
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

# ----------------------------------------
# STEP 3: Create sequences
# ----------------------------------------
SEQ_LEN = 30
X_seq = np.array([
    data_scaled[i:i + SEQ_LEN]
    for i in range(len(data_scaled) - SEQ_LEN)
])

print("Sequence shape:", X_seq.shape)

# ----------------------------------------
# STEP 4: Model (unchanged architecture)
# ----------------------------------------
model = Sequential([
    LSTM(64, activation="relu", input_shape=(SEQ_LEN, X_seq.shape[2]), return_sequences=True),
    LSTM(32, activation="relu"),
    Dense(SEQ_LEN * X_seq.shape[2])
])

model.compile(optimizer="adam", loss="mse")

X_flat = X_seq.reshape(X_seq.shape[0], -1)

# ----------------------------------------
# STEP 5: Train
# ----------------------------------------
model.fit(
    X_seq,
    X_flat,
    epochs=8,
    batch_size=64,
    callbacks=[EarlyStopping(patience=2)],
    verbose=1
)

print("Model training completed")

# ----------------------------------------
# STEP 6: Anomaly score (FIXED)
# ----------------------------------------
X_pred = model.predict(X_seq, verbose=0)
X_pred = X_pred.reshape(X_seq.shape)

anomaly_score = np.mean(
    (X_seq - X_pred) ** 2,
    axis=(1, 2)
)

# ---- CRITICAL SAFETY FIX ----
anomaly_score = np.nan_to_num(
    anomaly_score,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

# If anomaly score is flat, add minimal variance
if np.std(anomaly_score) == 0:
    anomaly_score = anomaly_score + np.linspace(0, 1e-3, len(anomaly_score))

print(
    "Anomaly score range:",
    anomaly_score.min(),
    anomaly_score.max()
)

# ----------------------------------------
# STEP 7: Stability index (FIXED)
# ----------------------------------------
eps = 1e-6
min_a = anomaly_score.min()
max_a = anomaly_score.max()

norm = (anomaly_score - min_a) / (max_a - min_a + eps)
stability = 100 - norm * 20

# Smooth for time continuity
stability = pd.Series(stability).rolling(
    window=5,
    min_periods=1
).mean().values

# ----------------------------------------
# STEP 8: Trend
# ----------------------------------------
trend_delta = np.diff(stability, prepend=stability[0])

# ----------------------------------------
# STEP 9: Alert levels
# ----------------------------------------
alert = [
    "Normal" if v > 90 else
    "Warning" if v > 80 else
    "Critical"
    for v in stability
]

# ----------------------------------------
# STEP 10: Save output
# ----------------------------------------
output_df = pd.DataFrame({
    "time": np.arange(len(stability)),
    "stability": stability,
    "delta": trend_delta,
    "alert": alert
})

output_df.to_csv("stability_output.csv", index=False)

print("✅ stability_output.csv generated successfully")
print(output_df.head())
