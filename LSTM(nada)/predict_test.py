import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from data_preprocessor import prepare_fall_detection_data

# Assuming the scaler and best model have been saved by the training script
SCALER_PATH = "scaler.save"
MODEL_PATH = "best_lstm_fall_detector.h5"

# --- Hyperparameters (Must match training parameters) ---
TIME_STEPS = 50 
TEST_SIZE = 0.2
RANDOM_STATE = 42
STEP_SIZE = 2 # تم التعديل من 1 إلى 2 ليتطابق مع كود التدريب الجديد
# --------------------------------------------------------

def evaluate_model_performance():
    try:
        
        print("⏳ Re-preparing data to get X_test...")
        # Note: We only need X_test and y_test for evaluation
        _, X_test, _, y_test = prepare_fall_detection_data(
            filepath='DataSet.csv', 
            time_steps=TIME_STEPS, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE, 
            step_size=STEP_SIZE
        )

        print(f"Test sequences shape: {X_test.shape}")

        print("🧠 Loading the best trained model...")
        # Load the best model saved during training
        model = tf.keras.models.load_model(MODEL_PATH)

        print("🔮 Making predictions...")
        # Predict probabilities
        y_pred_probs = model.predict(X_test)
        
        # Convert probabilities to binary predictions (0 or 1)
        # 0.5 is the standard threshold for binary classification
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        
        print("\n--- 📊 Evaluation Results ---")
        
        # 1. Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Overall Accuracy: {accuracy*100:.2f}%")

        # 2. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        # Interpretation: [[True Negatives, False Positives], [False Negatives, True Positives]]

        # 3. Classification Report (Precision, Recall, F1-Score)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Fall (0)', 'Fall (1)']))

    except FileNotFoundError as e:
        print(f"🚨 Error: One of the required files was not found: {e}")
        print("Please ensure 'best_lstm_fall_detector.h5' and 'DataSet.csv' exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    evaluate_model_performance()