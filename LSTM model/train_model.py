import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import os

# --- CRITICAL IMPORT ---
# This requires a file named 'data_preprocessor.py' to exist in the same directory.
# This function provides the prepared X_train, X_test, y_train, y_test data.
from data_preprocessor import prepare_fall_detection_data
# --- END CRITICAL IMPORT ---


def build_lstm_model(time_steps, n_features, lstm_units=64, dropout_rate=0.2):
    """
    Defines and compiles a sequential LSTM model for binary classification.

    Args:
        time_steps (int): The sequence length (window size).
        n_features (int): The number of features (sensor axes).
        lstm_units (int): Number of units in the LSTM layer.
        dropout_rate (float): Dropout rate for regularization.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    model = Sequential([
        # LSTM layer: input shape is (TIME_STEPS, N_FEATURES)
        LSTM(units=lstm_units, input_shape=(time_steps, n_features), return_sequences=False),
        
        # Dropout layer for regularization to prevent overfitting
        Dropout(dropout_rate),
        
        # Output layer: 1 unit with sigmoid activation for binary classification (Fall/Not Fall)
        Dense(units=1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy', # Appropriate loss function for binary classification
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()] 
    )
    return model

def train_and_evaluate_model(X_train, X_test, y_train, y_test, epochs=20, batch_size=64):
    """
    Trains and evaluates the LSTM model, applying class weights to handle data imbalance,
    and saves the trained model.
    """
    if X_train is None or X_test is None:
        print("Error: Training data is not available. Please check data_preprocessor.py.")
        return

    # --- CRITICAL FOR IMBALANCE: Calculate Class Weights ---
    # Calculates weights to balance the minority class ('Fall' = 1) and the majority class ('Not Fall' = 0).
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    # Convert weights array to dictionary format: {0: weight_for_0, 1: weight_for_1}
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\nCalculated Class Weights: {class_weight_dict}")
    # --------------------------------------------------------

    # Extract dimensions from the training data
    TIME_STEPS = X_train.shape[1]
    N_FEATURES = X_train.shape[2]

    # Build the model
    model = build_lstm_model(TIME_STEPS, N_FEATURES)
    
    print("\n--- Model Architecture ---")
    model.summary()
    print("--------------------------")

    # Train the Model
    print("\n🚀 Starting Model Training (with Class Weights)...")
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1, # Use 10% of the training data for validation
        shuffle=False, 
        verbose=1,
        class_weight=class_weight_dict  # <--- Apply weights during training
    )

    print("\n✅ Training Complete.")

    # Evaluate the Model on the unseen test set
    print("\n--- Evaluating Model on Test Set ---")
    if X_test.shape[0] > 0:
        loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy*100:.2f}%")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
    else:
        print("Test set is empty, skipping evaluation.")

    # --- FINAL STEP: Model Saving ---
    model_filepath = 'fall_detection_model.h5'
    try:
        save_model(model, model_filepath)
        print(f"\n💾 Model saved successfully to {model_filepath}")
    except Exception as e:
        print(f"\nError saving model: {e}")
    # -------------------------------


if __name__ == '__main__':
    # 1. Prepare data (calls the function from data_preprocessor.py)
    print("--- Data Preprocessing Started ---")
    X_train, X_test, y_train, y_test = prepare_fall_detection_data(
        filepath='DataSet.csv', 
        time_steps=50, 
        test_size=0.2, 
        random_state=42
    )
    print("--- Data Preprocessing Complete ---")
    
    if X_train is not None:
        # 2. Train and evaluate the model
        train_and_evaluate_model(X_train, X_test, y_train, y_test, epochs=20, batch_size=64)