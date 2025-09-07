
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

class RandomForestModel:
    """
    A class for implementing and training a Random Forest Regressor model.
    """
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        """
        Initializes the Random Forest Regressor model.

        Args:
            n_estimators (int): The number of trees in the forest.
            max_depth (int): The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
            random_state (int): Controls the randomness of the bootstrapping of the samples
                                for each tree and the sampling of the features to consider
                                when looking for the best split at each node.
        """
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Trains the Random Forest Regressor model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
        """
        print("Training Random Forest Regressor model...")
        self.model.fit(X_train, y_train)
        print("Random Forest Regressor model trained.")

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """
        Makes predictions using the trained Random Forest Regressor model.

        Args:
            X_test (pd.DataFrame): Testing features.

        Returns:
            pd.Series: Predicted values.
        """
        print("Making predictions with Random Forest Regressor model...")
        predictions = self.model.predict(X_test)
        print("Predictions made.")
        return pd.Series(predictions, index=X_test.index)

if __name__ == '__main__':
    # Example usage (dummy data)
    from sklearn.model_selection import train_test_split
    import numpy as np

    # Generate some dummy data
    np.random.seed(42)
    data_size = 100
    X = pd.DataFrame(np.random.rand(data_size, 5), columns=[f'feature_{i}' for i in range(5)])
    y = pd.Series(2 * X['feature_0'] + 3 * X['feature_1'] + np.random.randn(data_size) * 0.1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestModel()
    rf_model.train(X_train, y_train)
    predictions = rf_model.predict(X_test)

    print("\nSample Predictions:")
    print(predictions.head())
    print("\nActual Values:")
    print(y_test.head())


