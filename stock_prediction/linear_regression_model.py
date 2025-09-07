
import pandas as pd
from sklearn.linear_model import LinearRegression

class LinearRegressionModel:
    """
    A class for implementing and training a Linear Regression model.
    """
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Trains the Linear Regression model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
        """
        print("Training Linear Regression model...")
        self.model.fit(X_train, y_train)
        print("Linear Regression model trained.")

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """
        Makes predictions using the trained Linear Regression model.

        Args:
            X_test (pd.DataFrame): Test features.

        Returns:
            pd.Series: Predicted values.
        """
        print("Making predictions with Linear Regression model...")
        predictions = self.model.predict(X_test)
        print("Predictions made.")
        return pd.Series(predictions, index=X_test.index)

if __name__ == '__main__':
    # Example usage (requires preprocessed data)
    from preprocessor import DataPreprocessor

    preprocessor = DataPreprocessor()
    df = preprocessor.load_data('data/SP500_daily.csv')

    if not df.empty:
        lagged_df = preprocessor.create_lag_features(df[['Close_^GSPC']])
        X_lr = lagged_df.drop(columns=['Close_^GSPC'])
        y_lr = lagged_df['Close_^GSPC']
        X_train_lr, X_test_lr, y_train_lr, y_test_lr = preprocessor.split_data(X_lr, y_lr)

        lr_model = LinearRegressionModel()
        lr_model.train(X_train_lr, y_train_lr)
        predictions_lr = lr_model.predict(X_test_lr)

        print("\nLinear Regression Predictions (first 5):\n", predictions_lr.head())
        print("Actual Values (first 5):\n", y_test_lr.head())


