import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# ------------------------------------
# 1. Load model
# ------------------------------------
model = load_model("best_bilstm_fall_detection_model.keras")
print("✅ Model loaded")

# ------------------------------------
# 2. Load scaler (same used in training)
# ------------------------------------
scaler = joblib.load("scaler.save")
print("✅ Scaler loaded")

# ------------------------------------
# 3. Load and fix dataset
# ------------------------------------
def load_and_prepare_data(filepath="DataSet.csv"):
    df = pd.read_csv(filepath, header=[0,1])
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    cols = df.columns.tolist()

    # Fix accelerometer columns
    if 'WristAccelerometer_x-axis (g)' in cols:
        i = cols.index('WristAccelerometer_x-axis (g)')
        df.rename(columns={
            cols[i+1]: 'WristAccelerometer_y-axis (g)',
            cols[i+2]: 'WristAccelerometer_z-axis (g)'
        }, inplace=True)

    # Fix gyro columns
    if 'WristAngularVelocity_x-axis (deg/s)' in cols:
        i = cols.index('WristAngularVelocity_x-axis (deg/s)')
        df.rename(columns={
            cols[i+1]: 'WristAngularVelocity_y-axis (deg/s)',
            cols[i+2]: 'WristAngularVelocity_z-axis (deg/s)'
        }, inplace=True)

    # Find Activity & Subject columns (lowercase safe)
    activity_col = [c for c in df.columns if c.lower().startswith('activity')][0]
    subject_col  = [c for c in df.columns if 'subject' in c.lower()][0]

    # Add magnitude features
    df['Acc_mag'] = np.sqrt(
        df['WristAccelerometer_x-axis (g)']**2 +
        df['WristAccelerometer_y-axis (g)']**2 +
        df['WristAccelerometer_z-axis (g)']**2
    )

    df['Gyro_mag'] = np.sqrt(
        df['WristAngularVelocity_x-axis (deg/s)']**2 +
        df['WristAngularVelocity_y-axis (deg/s)']**2 +
        df['WristAngularVelocity_z-axis (deg/s)']**2
    )

    return df, activity_col, subject_col

# ------------------------------------
# 4. Feature columns
# ------------------------------------
FEATURES = [
    'WristAccelerometer_x-axis (g)',
    'WristAccelerometer_y-axis (g)',
    'WristAccelerometer_z-axis (g)',
    'WristAngularVelocity_x-axis (deg/s)',
    'WristAngularVelocity_y-axis (deg/s)',
    'WristAngularVelocity_z-axis (deg/s)',
    'Acc_mag',
    'Gyro_mag'
]

# ------------------------------------
# 5. Load dataset
# ------------------------------------
df, activity_col, subject_col = load_and_prepare_data()

# ------------------------------------
# 6. Ask user for input
# ------------------------------------
subject = int(input("Enter Subject ID: "))
activity = int(input("Enter Activity ID: "))

# ------------------------------------
# 7. Filter data
# ------------------------------------
filtered = df[(df[subject_col] == subject) & (df[activity_col] == activity)]

if len(filtered) == 0:
    print("❌ No data found for this Subject & Activity")
    exit()

print(f"✅ Found {len(filtered)} rows")

# ------------------------------------
# 8. Prepare input for LSTM
# ------------------------------------
X = filtered[FEATURES].values

time_steps = 50
if len(X) < time_steps:
    print(f"❌ Not enough rows (need at least {time_steps})")
    exit()

# Take first window
X_seq = X[:time_steps].reshape(1, time_steps, len(FEATURES))

# Normalize using training scaler
X_scaled = scaler.transform(
    X_seq.reshape(-1, X_seq.shape[-1])
).reshape(1, time_steps, len(FEATURES))

# ------------------------------------
# 9. Predict
# ------------------------------------
prob = model.predict(X_scaled)[0][0]
threshold = 0.35

print(f"\n📊 Prediction probability: {prob:.4f}")

if prob >= threshold:
    print("🚨 Prediction: FALL")
else:
    print("✅ Prediction: NOT FALL")
