
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

class LSTMModel:
    """
    A class for implementing and training an LSTM Neural Network model.
    """
    def __init__(self, input_shape: tuple):
        self.model = self._build_model(input_shape)

    def _build_model(self, input_shape: tuple) -> Sequential:
        """
        Builds the LSTM model architecture.

        Args:
            input_shape (tuple): The shape of the input data (timesteps, features).

        Returns:
            Sequential: The compiled Keras Sequential model.
        """
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 50, batch_size: int = 32, validation_split: float = 0.1):
        """
        Trains the LSTM model.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training target.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            validation_split (float): Fraction of the training data to be used as validation data.
        """
        print("Training LSTM model...")
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[early_stopping], verbose=1)
        print("LSTM model trained.")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the trained LSTM model.

        Args:
            X_test (np.ndarray): Test features.

        Returns:
            np.ndarray: Predicted values.
        """
        print("Making predictions with LSTM model...")
        predictions = self.model.predict(X_test)
        print("Predictions made.")
        return predictions

if __name__ == '__main__':
    # Example usage (requires preprocessed data)
    from preprocessor import DataPreprocessor

    preprocessor = DataPreprocessor()
    df = preprocessor.load_data('data/SP500_daily.csv')

    if not df.empty:
        scaled_data = preprocessor.normalize_data(df['Close_^GSPC'])
        X_lstm, y_lstm = preprocessor.prepare_lstm_data(scaled_data)

        # Reshape X_lstm for LSTM input (samples, timesteps, features)
        X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

        split_index_lstm = int(len(X_lstm) * 0.8)
        X_train_lstm, X_test_lstm = X_lstm[:split_index_lstm], X_lstm[split_index_lstm:]
        y_train_lstm, y_test_lstm = y_lstm[:split_index_lstm], y_lstm[split_index_lstm:]

        lstm_model = LSTMModel(input_shape=(X_train_lstm.shape[1], 1))
        lstm_model.train(X_train_lstm, y_train_lstm)
        predictions_lstm = lstm_model.predict(X_test_lstm)

        print("\nLSTM Predictions (first 5):\n", predictions_lstm[:5].flatten())
        print("Actual Values (first 5):\n", y_test_lstm[:5].flatten())


