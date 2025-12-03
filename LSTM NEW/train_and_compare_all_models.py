import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ==============================
# Preprocessing مشترك
# ==============================
def prepare_common_data(filepath='DataSet.csv', time_steps=50, test_size=0.2, step_size=2):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(filepath, low_memory=False)

    # Extract wrist data
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

    # Magnitudes
    df['Acc_mag'] = np.sqrt(df['WristAccelerometer_x']**2 +
                            df['WristAccelerometer_y']**2 +
                            df['WristAccelerometer_z']**2)
    df['Gyro_mag'] = np.sqrt(df['WristAngularVelocity_x']**2 +
                             df['WristAngularVelocity_y']**2 +
                             df['WristAngularVelocity_z']**2)
    FEATURES += ['Acc_mag','Gyro_mag']

    # fall_now
    FALL_CODES = [7,8,9,10,11]
    df['fall_now'] = df['Activity'].apply(lambda x: 1 if x in FALL_CODES else 0)

    # fall_soon
    HORIZON = 10
    fall_series = df['fall_now'].values
    df['fall_soon'] = [int(fall_series[i+1:i+HORIZON+1].max()) if i+HORIZON < len(fall_series) else 0
                       for i in range(len(fall_series))]

    # Fill missing
    df[FEATURES] = df[FEATURES].fillna(0)

    # Split train/test
    X = df[FEATURES].values
    y_now = df['fall_now'].values
    y_soon = df['fall_soon'].values

    X_train_raw, X_test_raw, y_now_train, y_now_test, y_soon_train, y_soon_test = train_test_split(
        X, y_now, y_soon, test_size=test_size, random_state=42, stratify=y_now
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    joblib.dump(scaler, "scaler_all.save")

    # Sliding window
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

    return X_train, X_test, y_now_train_seq, y_now_test_seq, y_soon_train_seq, y_soon_test_seq

# ==============================
# LSTM + Attention
# ==============================
def attention_block(inputs):
    score = layers.Dense(128, activation='tanh')(inputs)
    score = layers.Dense(1, activation='sigmoid')(score)  # ✅ sigmoid instead of softmax
    attention = layers.Multiply()([inputs, score])
    return layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(attention)

def build_lstm_attention(time_steps, features):
    inp = Input(shape=(time_steps, features))
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inp)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x2 = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Add()([x, x2])
    x = layers.LayerNormalization()(x)
    att = attention_block(x)
    shared = layers.Dense(128, activation='relu')(att)
    shared = layers.Dropout(0.3)(shared)
    fall_now = layers.Dense(64, activation='relu')(shared)
    fall_now = layers.Dense(1, activation='sigmoid', name='fall_now')(fall_now)
    fall_soon = layers.Dense(64, activation='relu')(shared)
    fall_soon = layers.Dense(1, activation='sigmoid', name='fall_soon')(fall_soon)
    model = Model(inp, [fall_now, fall_soon])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0008),
        loss='binary_crossentropy',
        metrics={
            "fall_now": ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
            "fall_soon": ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        }
    )
    return model

# ==============================
# CNN + LSTM
# ==============================
def build_cnn_lstm(time_steps, features):
    inp = Input(shape=(time_steps, features))
    x = layers.Conv1D(64,3,activation='relu',padding='same')(inp)
    x = layers.Conv1D(64,3,activation='relu',padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(128))(x)
    x = layers.Dropout(0.3)(x)
    shared = layers.Dense(128, activation='relu')(x)
    shared = layers.Dropout(0.3)(shared)
    fall_now = layers.Dense(64, activation='relu')(shared)
    fall_now = layers.Dense(1, activation='sigmoid', name='fall_now')(fall_now)
    fall_soon = layers.Dense(64, activation='relu')(shared)
    fall_soon = layers.Dense(1, activation='sigmoid', name='fall_soon')(fall_soon)
    model = Model(inp, [fall_now, fall_soon])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0008),
        loss='binary_crossentropy',
        metrics={
            "fall_now": ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
            "fall_soon": ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        }
    )
    return model

# ==============================
# Transformer + LSTM
# ==============================
def transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.2):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(x,x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dense(ff_dim, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1], activation='relu')(x)
    return x + res

def build_transformer_lstm(time_steps, features):
    inp = Input(shape=(time_steps, features))
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inp)
    x = layers.Dropout(0.2)(x)
    x = transformer_encoder(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dropout(0.3)(x)
    shared = layers.Dense(128, activation='relu')(x)
    shared = layers.Dropout(0.3)(shared)
    fall_now = layers.Dense(64, activation='relu')(shared)
    fall_now = layers.Dense(1, activation='sigmoid', name='fall_now')(fall_now)
    fall_soon = layers.Dense(64, activation='relu')(shared)
    fall_soon = layers.Dense(1, activation='sigmoid', name='fall_soon')(fall_soon)
    model = Model(inp, [fall_now, fall_soon])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0007),
        loss='binary_crossentropy',
        metrics={
            "fall_now": ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
            "fall_soon": ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        }
    )
    return model

# ==============================
# Training + Comparison
# ==============================
def train_and_compare():
    X_train, X_test, y_train_now, y_test_now, y_train_soon, y_test_soon = prepare_common_data()

    MODELS = [
        {"name":"LSTM_Attention","builder":build_lstm_attention},
        {"name":"CNN_LSTM","builder":build_cnn_lstm},
        {"name":"Transformer_LSTM","builder":build_transformer_lstm}
    ]

    results = []

    for m in MODELS:
        print(f"\n=== Training {m['name']} ===")
        model = m["builder"](X_train.shape[1], X_train.shape[2])
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=4),
            ModelCheckpoint(f"BEST_{m['name']}.keras", save_best_only=True, monitor='val_loss')
        ]
        history = model.fit(
            X_train, {"fall_now":y_train_now,"fall_soon":y_train_soon},
            validation_data=(X_test, {"fall_now":y_test_now,"fall_soon":y_test_soon}),
            epochs=60, batch_size=128, callbacks=callbacks, verbose=1
        )
        model.save(f"FINAL_{m['name']}.keras")

        y_pred_now, y_pred_soon = model.predict(X_test)
        y_pred_now = (y_pred_now>0.5).astype(int)
        y_pred_soon = (y_pred_soon>0.5).astype(int)

        metrics_now = {
            "Accuracy": accuracy_score(y_test_now, y_pred_now),
            "Precision": precision_score(y_test_now, y_pred_now),
            "Recall": recall_score(y_test_now, y_pred_now),
            "F1": f1_score(y_test_now, y_pred_now)
        }
        metrics_soon = {
            "Accuracy": accuracy_score(y_test_soon, y_pred_soon),
            "Precision": precision_score(y_test_soon, y_pred_soon),
            "Recall": recall_score(y_test_soon, y_pred_soon),
            "F1": f1_score(y_test_soon, y_pred_soon)
        }

        results.append({"model":m["name"], "fall_now":metrics_now, "fall_soon":metrics_soon})

        # Confusion Matrices
        ConfusionMatrixDisplay(confusion_matrix(y_test_now, y_pred_now)).plot(cmap="Blues")
        plt.title(f"{m['name']} - Fall Now")
        plt.show()
        ConfusionMatrixDisplay(confusion_matrix(y_test_soon, y_pred_soon)).plot(cmap="Oranges")
        plt.title(f"{m['name']} - Fall Soon")
        plt.show()

    # Bar chart comparison
    metrics_list = ["Accuracy","Precision","Recall","F1"]
    for metric in metrics_list:
        plt.figure(figsize=(10,5))
        x = [r["model"] for r in results]
        y_now = [r["fall_now"][metric] for r in results]
        y_soon = [r["fall_soon"][metric] for r in results]
        plt.bar([i-0.15 for i in range(len(x))], y_now, width=0.3, label="Fall Now")
        plt.bar([i+0.15 for i in range(len(x))], y_soon, width=0.3, label="Fall Soon")
        plt.xticks(range(len(x)), x)
        plt.ylabel(metric)
        plt.title(f"Comparison of {metric}")
        plt.legend()
        plt.ylim(0,1)
        plt.show()

if __name__=="__main__":
    train_and_compare()
