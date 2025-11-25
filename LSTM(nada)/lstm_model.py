import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_lstm_model(input_shape, lstm_units=128, dropout_rate=0.5):
    """
    Creates a Deep Sequential LSTM model (2 layers) for higher accuracy.
    """
    model = Sequential([
        # الطبقة الأولى: يجب أن تعيد التسلسل (return_sequences=True)
        LSTM(lstm_units, input_shape=input_shape, return_sequences=True), 
        Dropout(dropout_rate),

        # الطبقة الثانية: تعالج التسلسل الذي أرجعته الطبقة الأولى
        LSTM(lstm_units), 
        Dropout(dropout_rate),
        
        # طبقة كثيفة إضافية
        Dense(64, activation='relu'), 
        Dropout(dropout_rate),
        
        # Output layer
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

if __name__ == '__main__':
    example_model = create_lstm_model(input_shape=(50, 8))
    example_model.summary()