
import pandas as pd
import numpy as np
from data_collector import download_data
from preprocessor import DataPreprocessor
from linear_regression_model import LinearRegressionModel
from lstm_model import LSTMModel
from random_forest_model import RandomForestModel
from evaluator import ModelEvaluator
import os

def main():
    """
    Main function to run the stock price prediction pipeline.
    It orchestrates data collection, preprocessing, model training,
    prediction, and evaluation for Linear Regression, LSTM, and Random Forest models.
    """
    # --- Master Configuration ---
    TICKER_SYMBOL = '^GSPC'  # S&P 500 Index
    TRAIN_START_DATE = '2015-01-01'
    TRAIN_END_DATE = '2024-12-31'
    EVAL_START_DATE = '2025-01-01'
    EVAL_END_DATE = '2025-06-30'
    LAG_STEPS = 5
    TEST_SIZE = 0.2
    LSTM_EPOCHS = 50
    LSTM_BATCH_SIZE = 32
    RF_N_ESTIMATORS = 200
    RF_MAX_DEPTH = 10

    # --- Dynamic File Paths ---
    train_raw_data_path = f'data/{TICKER_SYMBOL}_train_data.csv'
    eval_raw_data_path = f'data/{TICKER_SYMBOL}_eval_data.csv'

    # Initialize modules
    preprocessor = DataPreprocessor()
    evaluator = ModelEvaluator(output_dir="./reports", plot_dir="./plots")

    # 1. Data Collection for Training/Testing
    print(f"\n--- Starting Data Collection for {TICKER_SYMBOL} (Training/Testing: {TRAIN_START_DATE} to {TRAIN_END_DATE}) ---")
    download_data(TICKER_SYMBOL, TRAIN_START_DATE, TRAIN_END_DATE, train_raw_data_path)
    df_train = preprocessor.load_data(train_raw_data_path)

    if df_train.empty:
        print("Exiting: No training data to process.")
        return

    # 2. Preprocessing for Linear Regression and Random Forest
    print("\n--- Preprocessing Data for Linear Regression and Random Forest ---")
    X_lr_rf, y_lr_rf = preprocessor.create_features_and_target(df_train, lag_steps=LAG_STEPS)
    X_train_lr_rf, X_test_lr_rf, y_train_lr_rf, y_test_lr_rf = preprocessor.split_data_chronologically(X_lr_rf, y_lr_rf, TEST_SIZE)

    # Get the actual previous day's close prices for LR/RF test set for plotting
    # This needs to be from the original df_train, aligned with y_test_lr_rf
    original_df_train_aligned = df_train.loc[y_test_lr_rf.index].shift(1)[preprocessor.close_column]
    original_df_train_aligned.dropna(inplace=True)
    # Adjust y_test_lr_rf and X_test_lr_rf to match the length of original_df_train_aligned
    y_test_lr_rf = y_test_lr_rf.loc[original_df_train_aligned.index]
    X_test_lr_rf = X_test_lr_rf.loc[original_df_train_aligned.index]

    # 3. Linear Regression Model Training
    print("\n--- Training Linear Regression Model ---")
    lr_model = LinearRegressionModel()
    lr_model.train(X_train_lr_rf, y_train_lr_rf)
    predictions_lr_test_returns = lr_model.predict(X_test_lr_rf)
    predictions_lr_test_prices = original_df_train_aligned * (1 + predictions_lr_test_returns)
    y_test_lr_rf_prices = original_df_train_aligned * (1 + y_test_lr_rf)

    # 4. Random Forest Model Training
    print("\n--- Training Random Forest Regressor Model ---")
    rf_model = RandomForestModel(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH)
    rf_model.train(X_train_lr_rf, y_train_lr_rf)
    predictions_rf_test_returns = rf_model.predict(X_test_lr_rf)
    predictions_rf_test_prices = original_df_train_aligned * (1 + predictions_rf_test_returns)

    # 5. Preprocessing for LSTM (Training/Testing)
    print("\n--- Preprocessing Data for LSTM Neural Network (Training/Testing) ---")
    # For LSTM, we predict the next day's price based on previous 'lag_steps' prices.
    # The target for LSTM is implicitly handled by prepare_lstm_data and then aligned with inverse_normalize_data.
    scaled_data_lstm = preprocessor.normalize_data(df_train[preprocessor.close_column])
    X_lstm, y_lstm = preprocessor.prepare_lstm_data(scaled_data_lstm, lag_steps=LAG_STEPS)

    # Reshape X_lstm for LSTM input (samples, timesteps, features)
    X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

    # Split LSTM data chronologically
    split_index_lstm = int(len(X_lstm) * (1 - TEST_SIZE))
    X_train_lstm, X_test_lstm = X_lstm[:split_index_lstm], X_lstm[split_index_lstm:]
    y_train_lstm, y_test_lstm = y_lstm[:split_index_lstm], y_lstm[split_index_lstm:]

    # 6. LSTM Neural Network Model Training
    print("\n--- Training LSTM Neural Network Model ---")
    lstm_model = LSTMModel(input_shape=(X_train_lstm.shape[1], 1))
    lstm_model.train(X_train_lstm, y_train_lstm, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE)
    predictions_lstm_test_scaled = lstm_model.predict(X_test_lstm)

    # Inverse transform LSTM test predictions to original scale
    predictions_lstm_test = preprocessor.inverse_normalize_data(predictions_lstm_test_scaled)
    y_test_lstm_original = preprocessor.inverse_normalize_data(y_test_lstm.reshape(-1, 1))

    # 7. Evaluation on Test Set
    print("\n--- Evaluating Models on Test Set ---")
    lr_metrics_test = evaluator.evaluate_model(y_test_lr_rf_prices.values, predictions_lr_test_prices.values, f"Linear Regression (Test Set: {TICKER_SYMBOL})")
    rf_metrics_test = evaluator.evaluate_model(y_test_lr_rf_prices.values, predictions_rf_test_prices.values, f"Random Forest (Test Set: {TICKER_SYMBOL})")
    lstm_metrics_test = evaluator.evaluate_model(y_test_lstm_original, predictions_lstm_test, f"LSTM Neural Network (Test Set: {TICKER_SYMBOL})")

    # 8. Visualization on Test Set
    print("\n--- Generating Plots for Test Set ---")
    evaluator.plot_predictions(y_test_lr_rf_prices.values, predictions_lr_test_prices.values, f"Linear Regression: {TICKER_SYMBOL} Test Set ({TRAIN_START_DATE} to {TRAIN_END_DATE})", f"{TICKER_SYMBOL}_lr_test_predictions.png")
    evaluator.plot_predictions(y_test_lr_rf_prices.values, predictions_rf_test_prices.values, f"Random Forest: {TICKER_SYMBOL} Test Set ({TRAIN_START_DATE} to {TRAIN_END_DATE})", f"{TICKER_SYMBOL}_rf_test_predictions.png")
    evaluator.plot_predictions(y_test_lstm_original, predictions_lstm_test, f"LSTM Neural Network: {TICKER_SYMBOL} Test Set ({TRAIN_START_DATE} to {TRAIN_END_DATE})", f"{TICKER_SYMBOL}_lstm_test_predictions.png")

    # --- Out-of-Sample Evaluation ---

    # 9. Data Collection for Evaluation Period
    print(f"\n--- Starting Data Collection for {TICKER_SYMBOL} (Evaluation Period: {EVAL_START_DATE} to {EVAL_END_DATE}) ---")
    download_data(TICKER_SYMBOL, EVAL_START_DATE, EVAL_END_DATE, eval_raw_data_path)
    df_eval = preprocessor.load_data(eval_raw_data_path)

    if df_eval.empty:
        print("Exiting: No evaluation data to process.")
        return

    # 10. Preprocessing for Linear Regression and Random Forest (Evaluation Period)
    print("\n--- Preprocessing Data for Linear Regression and Random Forest (Evaluation Period) ---")
    X_lr_rf_eval, y_lr_rf_eval = preprocessor.create_features_and_target(df_eval, lag_steps=LAG_STEPS)

    # Get the actual previous day's close prices for LR/RF eval set for plotting
    original_df_eval_aligned = df_eval.loc[y_lr_rf_eval.index].shift(1)[preprocessor.close_column]
    original_df_eval_aligned.dropna(inplace=True)
    # Adjust y_lr_rf_eval and X_lr_rf_eval to match the length of original_df_eval_aligned
    y_lr_rf_eval = y_lr_rf_eval.loc[original_df_eval_aligned.index]
    X_lr_rf_eval = X_lr_rf_eval.loc[original_df_eval_aligned.index]

    # 11. Linear Regression Model Prediction on Evaluation Period
    print("\n--- Making Predictions with Linear Regression Model on Evaluation Period ---")
    predictions_lr_eval_returns = lr_model.predict(X_lr_rf_eval)
    predictions_lr_eval_prices = original_df_eval_aligned * (1 + predictions_lr_eval_returns)
    y_lr_rf_eval_prices = original_df_eval_aligned * (1 + y_lr_rf_eval)

    # 12. Random Forest Model Prediction on Evaluation Period
    print("\n--- Making Predictions with Random Forest Model on Evaluation Period ---")
    predictions_rf_eval_returns = rf_model.predict(X_lr_rf_eval)
    predictions_rf_eval_prices = original_df_eval_aligned * (1 + predictions_rf_eval_returns)

    # 13. Preprocessing for LSTM (Evaluation Period)
    print("\n--- Preprocessing Data for LSTM Neural Network (Evaluation Period) ---")
    scaled_data_lstm_eval = preprocessor.normalize_data(df_eval[preprocessor.close_column])
    X_lstm_eval, y_lstm_eval = preprocessor.prepare_lstm_data(scaled_data_lstm_eval, lag_steps=LAG_STEPS)

    # Reshape X_lstm_eval for LSTM input (samples, timesteps, features)
    X_lstm_eval = np.reshape(X_lstm_eval, (X_lstm_eval.shape[0], X_lstm_eval.shape[1], 1))

    # 14. LSTM Neural Network Model Prediction on Evaluation Period
    print("\n--- Making Predictions with LSTM Neural Network Model on Evaluation Period ---")
    predictions_lstm_eval_scaled = lstm_model.predict(X_lstm_eval)

    # Inverse transform LSTM evaluation predictions to original scale
    predictions_lstm_eval = preprocessor.inverse_normalize_data(predictions_lstm_eval_scaled)
    y_eval_lstm_original = preprocessor.inverse_normalize_data(y_lstm_eval.reshape(-1, 1))

    # 15. Evaluation on Evaluation Set
    print("\n--- Evaluating Models on Evaluation Set ---")
    lr_metrics_eval = evaluator.evaluate_model(y_lr_rf_eval_prices.values, predictions_lr_eval_prices.values, f"Linear Regression (Evaluation Set: {TICKER_SYMBOL})")
    rf_metrics_eval = evaluator.evaluate_model(y_lr_rf_eval_prices.values, predictions_rf_eval_prices.values, f"Random Forest (Evaluation Set: {TICKER_SYMBOL})")
    lstm_metrics_eval = evaluator.evaluate_model(y_eval_lstm_original, predictions_lstm_eval, f"LSTM Neural Network (Evaluation Set: {TICKER_SYMBOL})")

    # 16. Visualization on Evaluation Set
    print("\n--- Generating Plots for Evaluation Set ---")
    evaluator.plot_predictions(y_lr_rf_eval_prices.values, predictions_lr_eval_prices.values, f"Linear Regression: {TICKER_SYMBOL} Evaluation Set ({EVAL_START_DATE} to {EVAL_END_DATE})", f"{TICKER_SYMBOL}_lr_eval_predictions.png")
    evaluator.plot_predictions(y_lr_rf_eval_prices.values, predictions_rf_eval_prices.values, f"Random Forest: {TICKER_SYMBOL} Evaluation Set ({EVAL_START_DATE} to {EVAL_END_DATE})", f"{TICKER_SYMBOL}_rf_eval_predictions.png")
    evaluator.plot_predictions(y_eval_lstm_original, predictions_lstm_eval, f"LSTM Neural Network: {TICKER_SYMBOL} Evaluation Set ({EVAL_START_DATE} to {EVAL_END_DATE})", f"{TICKER_SYMBOL}_lstm_eval_predictions.png")

    # 17. Generate Combined Report
    print("\n--- Generating Combined Performance Report ---")
    report_content = f"Stock Price Prediction Model Performance Report for {TICKER_SYMBOL}\n"
    report_content += "=======================================================\n\n"

    report_content += f"--- Test Set Performance (Data from {TRAIN_START_DATE} to {TRAIN_END_DATE}) ---\n"
    report_content += "Linear Regression (Test Set):\n"
    for metric, value in lr_metrics_test.items():
        report_content += f"  {metric}: {value:.4f}\n"
    report_content += "\n"
    report_content += "Random Forest (Test Set):\n"
    for metric, value in rf_metrics_test.items():
        report_content += f"  {metric}: {value:.4f}\n"
    report_content += "\n"
    report_content += "LSTM Neural Network (Test Set):\n"
    for metric, value in lstm_metrics_test.items():
        report_content += f"  {metric}: {value:.4f}\n"
    report_content += "\n"

    report_content += f"--- Evaluation Set Performance (Data from {EVAL_START_DATE} to {EVAL_END_DATE}) ---\n"
    report_content += "Linear Regression (Evaluation Set):\n"
    for metric, value in lr_metrics_eval.items():
        report_content += f"  {metric}: {value:.4f}\n"
    report_content += "\n"
    report_content += "Random Forest (Evaluation Set):\n"
    for metric, value in rf_metrics_eval.items():
        report_content += f"  {metric}: {value:.4f}\n"
    report_content += "\n"
    report_content += "LSTM Neural Network (Evaluation Set):\n"
    for metric, value in lstm_metrics_eval.items():
        report_content += f"  {metric}: {value:.4f}\n"
    report_content += "\n"

    report_content += "Note: Lower MSE/RMSE and higher R2 Score indicate better model performance.\n"

    report_filename = f"{TICKER_SYMBOL}_combined_performance_report.txt"
    report_path = os.path.join(evaluator.output_dir, report_filename)
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"Combined performance report saved to {report_path}")

    print("\n--- Stock Price Prediction Pipeline Completed ---")

if __name__ == "__main__":
    main()


