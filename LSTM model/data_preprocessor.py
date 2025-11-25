import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def prepare_fall_detection_data(filepath='DataSet.csv', time_steps=50, test_size=0.2, random_state=42, step_size=5):
    """
    Preprocess dataset for fall detection with magnitude features and sliding window.
    Returns: X_train, X_test, y_train, y_test
    """
    try:
        df = pd.read_csv(filepath, header=[0,1])
        df.columns = ['_'.join(col).strip() for col in df.columns.values] 
        cols = df.columns.tolist()

        # Fix accelerometer columns
        if 'WristAccelerometer_x-axis (g)' in cols:
            i = cols.index('WristAccelerometer_x-axis (g)')
            df.rename(columns={cols[i+1]:'WristAccelerometer_y-axis (g)', cols[i+2]:'WristAccelerometer_z-axis (g)'}, inplace=True)

        # Fix gyro columns
        if 'WristAngularVelocity_x-axis (deg/s)' in cols:
            i = cols.index('WristAngularVelocity_x-axis (deg/s)')
            df.rename(columns={cols[i+1]:'WristAngularVelocity_y-axis (deg/s)', cols[i+2]:'WristAngularVelocity_z-axis (deg/s)'}, inplace=True)

        # Find activity column
        activity_col = [c for c in df.columns if c.lower().startswith('activity')][0]

        FEATURES = [
            'WristAccelerometer_x-axis (g)', 'WristAccelerometer_y-axis (g)', 'WristAccelerometer_z-axis (g)',
            'WristAngularVelocity_x-axis (deg/s)', 'WristAngularVelocity_y-axis (deg/s)', 'WristAngularVelocity_z-axis (deg/s)'
        ]

        # Add magnitude features
        df['Acc_mag'] = np.sqrt(df['WristAccelerometer_x-axis (g)']**2 +
                                df['WristAccelerometer_y-axis (g)']**2 +
                                df['WristAccelerometer_z-axis (g)']**2)
        df['Gyro_mag'] = np.sqrt(df['WristAngularVelocity_x-axis (deg/s)']**2 +
                                 df['WristAngularVelocity_y-axis (deg/s)']**2 +
                                 df['WristAngularVelocity_z-axis (deg/s)']**2)
        FEATURES += ['Acc_mag','Gyro_mag']

        FALL_CODES = [7,8,9,10,11]
        df['Target'] = df[activity_col].apply(lambda x: 1 if x in FALL_CODES else 0)

        X = df[FEATURES].values
        y = df['Target'].values

        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)
        joblib.dump(scaler, "scaler.save")
        print("✅ Scaler saved")

        def create_sequences(data, target, steps, step_size=1):
            X_seq, y_seq = [], []
            for i in range(0, len(data)-steps, step_size):
                X_seq.append(data[i:i+steps])
                y_seq.append(target[i+steps-1])
            return np.array(X_seq), np.array(y_seq)

        X_train, y_train = create_sequences(X_train_scaled, y_train_raw, time_steps, step_size)
        X_test, y_test = create_sequences(X_test_scaled, y_test_raw, time_steps, step_size)

    except Exception as e:
        print(f"❌ Error preprocessing: {e}")
        return None, None, None, None

    print("✅ Preprocessing finished")
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    prepare_fall_detection_data()
