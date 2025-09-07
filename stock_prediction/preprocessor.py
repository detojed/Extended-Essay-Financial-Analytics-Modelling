
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    """
    A class for preprocessing stock market data, including normalization,
    feature creation, and train-test splitting.
    """
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.close_column = None # To store the dynamically identified close column name

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Loads data from a CSV file.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        try:
            df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            # Dynamically find the 'Close' column
            close_cols = [col for col in df.columns if col.startswith('Close_')]
            if close_cols:
                self.close_column = close_cols[0]
            else:
                # Fallback if no 'Close_' column is found, assume 'Close'
                if 'Close' in df.columns:
                    self.close_column = 'Close'
                else:
                    raise ValueError("No 'Close' column found in the data.")
            return df
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return pd.DataFrame()
        except ValueError as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()

    def normalize_data(self, data: pd.Series) -> np.ndarray:
        """
        Normalizes a single series of data using MinMaxScaler.

        Args:
            data (pd.Series): The data series to normalize.

        Returns:
            np.ndarray: Normalized data.
        """
        return self.scaler.fit_transform(data.values.reshape(-1, 1))

    def inverse_normalize_data(self, scaled_data: np.ndarray) -> np.ndarray:
        """
        Inverses the normalization on a scaled data array.

        Args:
            scaled_data (np.ndarray): The scaled data array.

        Returns:
            np.ndarray: Inverse transformed data.
        """
        return self.scaler.inverse_transform(scaled_data)

    def create_features_and_target(self, df: pd.DataFrame, lag_steps: int = 5) -> tuple[pd.DataFrame, pd.Series]:
        """
        Creates features and the corresponding target for the models.
        The target is the next day's daily return.

        Args:
            df (pd.DataFrame): The input DataFrame with stock data.
            lag_steps (int): Number of past days to use as lag features.

        Returns:
            tuple[pd.DataFrame, pd.Series]: Features DataFrame (X) and Target Series (y).
        """
        df_processed = df.copy()

        # Calculate daily returns
        df_processed["Daily_Return"] = df_processed[self.close_column].pct_change()

        # Create the target variable (next day's daily return)
        df_processed["Target_Daily_Return"] = df_processed["Daily_Return"].shift(-1)

        # Lag features for Daily_Return
        for i in range(1, lag_steps + 1):
            df_processed[f'Daily_Return_Lag_{i}'] = df_processed["Daily_Return"].shift(i)

        # Moving Averages of Close price (as features)
        df_processed["MA_5"] = df_processed[self.close_column].rolling(window=5).mean()
        df_processed["MA_10"] = df_processed[self.close_column].rolling(window=10).mean()

        # Drop rows with NaN values created by feature engineering and target shifting
        df_processed.dropna(inplace=True)

        # Define features (X) and target (y)
        # IMPORTANT: Do NOT include current day's Close price or Daily_Return in X to prevent data leakage.
        feature_columns = [col for col in df_processed.columns if col.startswith("Daily_Return_Lag_") or col.startswith("MA_")]
        X = df_processed[feature_columns]
        y = df_processed["Target_Daily_Return"]

        return X, y

    def split_data_chronologically(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> tuple:
        """
        Splits data into training and testing sets chronologically.

        Args:
            X (pd.DataFrame): Features DataFrame.
            y (pd.Series): Target Series.
            test_size (float): Proportion of the dataset to include in the test split.

        Returns:
            tuple: X_train, X_test, y_train, y_test.
        """
        split_index = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        return X_train, X_test, y_train, y_test

    def prepare_lstm_data(self, data: np.ndarray, lag_steps: int = 5) -> tuple:
        """
        Prepares data for LSTM model by creating sequences.

        Args:
            data (np.ndarray): Normalized data array (should be a single column).
            lag_steps (int): Number of past days to use as sequence length.

        Returns:
            tuple: X (sequences), y (targets).
        """
        X, y = [], []
        for i in range(len(data) - lag_steps):
            X.append(data[i:(i + lag_steps), 0])
            y.append(data[i + lag_steps, 0])
        return np.array(X), np.array(y)

if __name__ == '__main__':
    # Example usage:
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data('data/SP500_daily.csv')

    if not df.empty:
        # Preprocessing for Linear Regression and Random Forest
        X_lr_rf, y_lr_rf = preprocessor.create_features_and_target(df)
        X_train_lr_rf, X_test_lr_rf, y_train_lr_rf, y_test_lr_rf = preprocessor.split_data_chronologically(X_lr_rf, y_lr_rf)

        print("\nLinear Regression/Random Forest Data Shapes:")
        print(f"X_train_lr_rf shape: {X_train_lr_rf.shape}")
        print(f"y_train_lr_rf shape: {y_train_lr_rf.shape}")
        print(f"X_test_lr_rf shape: {X_test_lr_rf.shape}")
        print(f"y_test_lr_rf shape: {y_test_lr_rf.shape}")

        # Preprocessing for LSTM
        # For LSTM, we predict the next day's price based on previous 'lag_steps' prices.
        # So, we need to create the target within the LSTM preparation itself or ensure it's aligned.
        # Here, we'll just pass the close column and let prepare_lstm_data handle the sequence creation.
        scaled_data = preprocessor.normalize_data(df[preprocessor.close_column])
        X_lstm, y_lstm = preprocessor.prepare_lstm_data(scaled_data)

        # Split LSTM data chronologically
        split_index_lstm = int(len(X_lstm) * 0.8)
        X_train_lstm, X_test_lstm = X_lstm[:split_index_lstm], X_lstm[split_index_lstm:]
        y_train_lstm, y_test_lstm = y_lstm[:split_index_lstm], y_lstm[split_index_lstm:]

        print("\nLSTM Data Shapes:")
        print(f"X_train_lstm shape: {X_train_lstm.shape}")
        print(f"y_train_lstm shape: {y_train_lstm.shape}")
        print(f"X_test_lstm shape: {X_test_lstm.shape}")
        print(f"y_test_lstm shape: {y_test_lstm.shape}")


