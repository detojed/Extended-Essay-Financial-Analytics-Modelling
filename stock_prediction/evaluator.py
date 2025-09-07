
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import os

class ModelEvaluator:
    """
    A class for evaluating model performance and visualizing predictions.
    """
    def __init__(self, output_dir="./reports", plot_dir="./plots"):
        self.output_dir = output_dir
        self.plot_dir = plot_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
        """
        Calculates and prints evaluation metrics for a given model.

        Args:
            y_true (np.ndarray): Actual values.
            y_pred (np.ndarray): Predicted values.
            model_name (str): Name of the model being evaluated.

        Returns:
            dict: A dictionary containing MSE, RMSE, and R2 Score.
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        print(f"\n--- {model_name} Performance ---")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2 Score: {r2:.4f}")

        return {"MSE": mse, "RMSE": rmse, "R2 Score": r2}

    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, title: str, filename: str):
        """
        Generates and saves a plot of actual vs. predicted values.

        Args:
            y_true (np.ndarray): Actual values.
            y_pred (np.ndarray): Predicted values.
            title (str): Title of the plot.
            filename (str): Name of the file to save the plot.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label="Actual Prices")
        plt.plot(y_pred, label="Predicted Prices")
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(self.plot_dir, filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to {plot_path}")

    def generate_report(self, lr_metrics: dict, lstm_metrics: dict, report_filename: str = "performance_report.txt"):
        """
        Generates a performance report and saves it to a text file.

        Args:
            lr_metrics (dict): Metrics for Linear Regression model.
            lstm_metrics (dict): Metrics for LSTM Neural Network model.
            report_filename (str): Name of the file to save the report.
        """
        report_content = f"Stock Price Prediction Model Performance Report\n"
        report_content += "===============================================\n\n"

        report_content += "--- Linear Regression Performance ---\n"
        for metric, value in lr_metrics.items():
            report_content += f"  {metric}: {value:.4f}\n"
        report_content += "\n"

        report_content += "--- LSTM Neural Network Performance ---\n"
        for metric, value in lstm_metrics.items():
            report_content += f"  {metric}: {value:.4f}\n"
        report_content += "\n"

        report_content += "Note: Lower MSE/RMSE and higher R2 Score indicate better model performance.\n"

        report_path = os.path.join(self.output_dir, report_filename)
        with open(report_path, "w") as f:
            f.write(report_content)
        print(f"Performance report saved to {report_path}")


