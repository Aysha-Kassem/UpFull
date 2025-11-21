import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def prepare_fall_detection_data(filepath='DataSet.csv', time_steps=50, test_size=0.2, random_state=42):
    """
    Loads, cleans, scales, and sequences the time-series data for LSTM.
    Includes targeted fixes for inconsistent multi-level header naming and correct activity IDs.

    Args:
        filepath (str): Path to the CSV file.
        time_steps (int): The length of the sequence window for the LSTM input.
        test_size (float): The proportion of the data to use for the test set.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) numpy arrays.
    """
    try:
        # Load the CSV file, noting the multi-level header
        df = pd.read_csv(filepath, header=[0, 1])

        # Flatten the MultiIndex columns by joining the two levels, which still leaves 'Unnamed'
        # Example: ('WristAccelerometer', 'x-axis (g)') -> 'WristAccelerometer_x-axis (g)'
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
        
        # --- CRITICAL, TARGETED FIX FOR WRIST SENSOR COLUMNS ---
        # Rename the 'Unnamed' columns following the correctly named X-axis
        
        current_cols = df.columns.tolist()
        
        # 1. Fix WristAccelerometer columns (y and z axes)
        wrist_accel_x_col = 'WristAccelerometer_x-axis (g)'
        if wrist_accel_x_col in current_cols:
            x_index = current_cols.index(wrist_accel_x_col)
            
            # Rename the two columns immediately following the x-axis
            df.rename(columns={
                current_cols[x_index + 1]: 'WristAccelerometer_y-axis (g)',
                current_cols[x_index + 2]: 'WristAccelerometer_z-axis (g)',
            }, inplace=True)

        # 2. Fix WristAngularVelocity columns (y and z axes)
        wrist_gyro_x_col = 'WristAngularVelocity_x-axis (deg/s)'
        if wrist_gyro_x_col in current_cols:
            x_index = current_cols.index(wrist_gyro_x_col)
            
            # Rename the two columns immediately following the x-axis
            df.rename(columns={
                current_cols[x_index + 1]: 'WristAngularVelocity_y-axis (deg/s)',
                current_cols[x_index + 2]: 'WristAngularVelocity_z-axis (deg/s)',
            }, inplace=True)

        # ---------------------------------------------------------
        
        # Find Activity Column 
        activity_col = [col for col in df.columns if col.startswith('Activity')][0]

        # Define the 6 core motion features using the now corrected names
        FEATURE_COLS = [
            'WristAccelerometer_x-axis (g)',
            'WristAccelerometer_y-axis (g)',
            'WristAccelerometer_z-axis (g)',
            'WristAngularVelocity_x-axis (deg/s)',
            'WristAngularVelocity_y-axis (deg/s)',
            'WristAngularVelocity_z-axis (deg/s)'
        ]
        
        # --- CORRECT FALL Activity IDs based on user's input ---
        FALL_ACTIVITY_CODES = [7, 8, 9, 10, 11] 
        # -----------------------------------------------------

        # Create the binary target column: 1 for Fall, 0 for Not Fall
        df['Target'] = df[activity_col].apply(lambda x: 1 if x in FALL_ACTIVITY_CODES else 0)

    except Exception as e:
        # Final check for missing features after the targeted rename
        missing_features = [col for col in FEATURE_COLS if col not in df.columns]
        if missing_features:
            print(f"FATAL ERROR: The following expected feature columns are still missing: {missing_features}")
            print(f"Available columns: {list(df.columns)}")
        else:
            print(f"Error loading or preparing data: {e}")
        return None, None, None, None

    print(f"Data loaded. Total samples: {len(df)}")
    print(f"Selected Features: {len(FEATURE_COLS)} (6 motion axes)")

    # 1. Scaling the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLS].values)
    y_target = df['Target'].values

    # 2. Sequencing function
    def create_sequences(data, target, time_steps):
        """Transforms data into sequences (windows) suitable for LSTM."""
        X, y = [], []
        # Create sequences of TIME_STEPS length
        for i in range(len(data) - time_steps):
            X.append(data[i:(i + time_steps), :])
            # The label corresponds to the activity at the end of the sequence
            y.append(target[i + time_steps - 1]) 
        return np.array(X), np.array(y)

    # 3. Create the sequences
    X_sequences, y_labels = create_sequences(X_scaled, y_target, time_steps)
    
    print(f"Data scaled and sequenced. X shape: {X_sequences.shape}")

    # 4. Split data into training and test sets, maintaining the proportion of classes (stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X_sequences, 
        y_labels, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y_labels
    )
    
    print(f"Data split: X_train shape: {X_train.shape}")
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # Example usage when running this file directly
    X_train, X_test, y_train, y_test = prepare_fall_detection_data()
    if X_train is not None:
        print("\nPreprocessing complete. Ready to train model.")