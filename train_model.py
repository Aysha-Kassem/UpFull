import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from data_preprocessor import prepare_fall_detection_data 
from lstm_model import create_lstm_model 
from sklearn.model_selection import train_test_split # تأكد من أن هذا موجود في data_preprocessor 
from sklearn.preprocessing import StandardScaler # تأكد من أن هذا موجود في data_preprocessor

# --- Hyperparameters (Optimized for 90%+ Balanced Accuracy) ---
TIME_STEPS = 50 
STEP_SIZE = 2      # تم التعديل
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 50 
BATCH_SIZE = 64    # تم التعديل
LSTM_UNITS = 128   # تم التعديل
DROPOUT_RATE = 0.5 # تم التعديل
# -----------------------

def train_lstm_for_fall_detection():
    print("⏳ Preparing data...")
    X_train, X_test, y_train, y_test = prepare_fall_detection_data(
        filepath='DataSet.csv', 
        time_steps=TIME_STEPS, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        step_size=STEP_SIZE
    )

    print(f"Train sequences shape: {X_train.shape}")
    print(f"Test sequences shape: {X_test.shape}")
    
    # --- إضافة حساب ترجيح الفئة (Class Weights) ---
    # يهدف إلى معالجة اختلال الفئة وتحسين الـ Precision والـ Recall
    unique_classes, counts = np.unique(y_train, return_counts=True)
    class_totals = dict(zip(unique_classes, counts))
    
    neg = class_totals.get(0, 0) # عدد حالات عدم السقوط (No Fall)
    pos = class_totals.get(1, 0) # عدد حالات السقوط (Fall)
    total = len(y_train)

    # حساب الوزن بطريقة Inverse Class Frequency
    weight_for_0 = (1 / neg) * (total / 2.0) if neg > 0 else 1.0
    weight_for_1 = (1 / pos) * (total / 2.0) if pos > 0 else 1.0

    # زيادة وزن فئة No Fall (0) بـ 1.2 مرة لتحسين الـ Recall وتقليل False Positives
    class_weight = {0: weight_for_0 * 1.2, 1: weight_for_1} 
    print(f"Class Weights: {class_weight}")
    # -----------------------------------------------

    input_shape = (TIME_STEPS, X_train.shape[2])
    
    print("🧠 Creating LSTM model...")
    model = create_lstm_model(
        input_shape=input_shape,
        lstm_units=LSTM_UNITS,
        dropout_rate=DROPOUT_RATE
    )
    model.summary()
    
    # Callbacks for better training and preventing overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5, # تم التعديل
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        'best_lstm_fall_detector.h5', 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max', 
        verbose=1
    )
    
    print(f"🚀 Starting training for {EPOCHS} epochs...")
    history = model.fit(
        X_train, 
        y_train, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        validation_data=(X_test, y_test), 
        callbacks=[early_stopping, model_checkpoint],
        class_weight=class_weight, # **الخطوة الأهم لضبط الـ Metrics**
        verbose=1
    )
    
    # Evaluate the best model on the test set
    best_model = tf.keras.models.load_model('best_lstm_fall_detector.h5')
    loss, accuracy = best_model.evaluate(X_test, y_test, verbose=0)
    
    print("\n--- ✅ Training Complete ---")
    print(f"Final Test Accuracy (Best Model): {accuracy*100:.2f}%")
    print(f"Model saved as 'best_lstm_fall_detector.h5'")

if __name__ == '__main__':
    train_lstm_for_fall_detection()