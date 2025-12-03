# ============================================
# data_preprocessor_transformer.py
# ============================================
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def prepare_transformer_data(filepath='DataSet.csv', time_steps=50, test_size=0.2,
                             random_state=42, step_size=2):
    try:
        df = pd.read_csv(filepath, low_memory=False)

        # -------------------------------
        # 🟦 استخراج بيانات اليد فقط من الأعمدة
        # -------------------------------
        df['WristAccelerometer_x'] = pd.to_numeric(df['WristAccelerometer'], errors='coerce')
        df['WristAccelerometer_y'] = pd.to_numeric(df['Unnamed: 30'], errors='coerce')
        df['WristAccelerometer_z'] = pd.to_numeric(df['Unnamed: 31'], errors='coerce')

        df['WristAngularVelocity_x'] = pd.to_numeric(df['WristAngularVelocity'], errors='coerce')
        df['WristAngularVelocity_y'] = pd.to_numeric(df['Unnamed: 33'], errors='coerce')
        df['WristAngularVelocity_z'] = pd.to_numeric(df['Unnamed: 34'], errors='coerce')

        FEATURES = [
            'WristAccelerometer_x','WristAccelerometer_y','WristAccelerometer_z',
            'WristAngularVelocity_x','WristAngularVelocity_y','WristAngularVelocity_z'
        ]

        # -------------------------------
        # 🟦 إضافة Magnitude
        # -------------------------------
        df['Acc_mag'] = np.sqrt(
            df['WristAccelerometer_x']**2 +
            df['WristAccelerometer_y']**2 +
            df['WristAccelerometer_z']**2
        )
        df['Gyro_mag'] = np.sqrt(
            df['WristAngularVelocity_x']**2 +
            df['WristAngularVelocity_y']**2 +
            df['WristAngularVelocity_z']**2
        )
        FEATURES += ['Acc_mag','Gyro_mag']

        # -------------------------------
        # 🟥 fall_now
        # -------------------------------
        FALL_CODES = [7,8,9,10,11]
        df['fall_now'] = df['Activity'].apply(lambda x: 1 if x in FALL_CODES else 0)

        # -------------------------------
        # 🟧 fall_soon (within next 10 steps)
        # -------------------------------
        HORIZON = 10
        fall_series = df['fall_now'].values
        fall_soon = [int(fall_series[i+1:i+HORIZON+1].max()) if i+HORIZON < len(fall_series) else 0
                     for i in range(len(fall_series))]
        df['fall_soon'] = fall_soon

        # -------------------------------
        # ✅ Fill missing
        # -------------------------------
        df[FEATURES] = df[FEATURES].fillna(0)

        # -------------------------------
        # Split Train/Test
        # -------------------------------
        X = df[FEATURES].values
        y_now = df['fall_now'].values
        y_soon = df['fall_soon'].values

        X_train_raw, X_test_raw, y_now_train, y_now_test, y_soon_train, y_soon_test = train_test_split(
            X, y_now, y_soon, test_size=test_size, random_state=random_state, stratify=y_now
        )

        # -------------------------------
        # Scaling
        # -------------------------------
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)
        joblib.dump(scaler, "scaler_transformer.save")

        # -------------------------------
        # Sliding Window
        # -------------------------------
        def create_sequences(data, target, steps, step_size=1):
            X_seq, y_seq = [], []
            for i in range(0, len(data)-steps, step_size):
                X_seq.append(data[i:i+steps])
                y_seq.append(target[i+steps-1])
            return np.array(X_seq), np.array(y_seq)

        X_train, y_now_train_seq = create_sequences(X_train_scaled, y_now_train, time_steps, step_size)
        _, y_soon_train_seq = create_sequences(X_train_scaled, y_soon_train, time_steps, step_size)
        X_test, y_now_test_seq = create_sequences(X_test_scaled, y_now_test, time_steps, step_size)
        _, y_soon_test_seq = create_sequences(X_test_scaled, y_soon_test, time_steps, step_size)

        print("✅ Preprocessing finished")
        print("Train:", X_train.shape, "Test:", X_test.shape)
        return X_train, X_test, y_now_train_seq, y_now_test_seq, y_soon_train_seq, y_soon_test_seq

    except Exception as e:
        print(f"❌ Error preprocessing: {e}")
        return None, None, None, None, None, None

if __name__ == "__main__":
    prepare_transformer_data()
