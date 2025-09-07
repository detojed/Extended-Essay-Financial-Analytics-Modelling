# Stock Price Prediction Project

This project implements and compares three machine learning models—Linear Regression, LSTM Neural Networks, and Random Forest Regressors—for predicting short-term stock prices. It is designed to support an IB Computer Science Extended Essay project, prioritizing readability, modularity, and academic clarity.

## Features

- **Data Collection**: Downloads historical daily closing prices for a specified stock ticker (e.g., AAPL, ^GSPC) using the `yfinance` library.
- **Preprocessing**: Normalizes data (for LSTM), creates lag features, calculates daily returns and moving averages, and chronologically splits data into training, testing, and out-of-sample evaluation sets.
- **Model Implementation**: Includes:
    - **Linear Regression**: A basic linear regression model from `scikit-learn`.
    - **LSTM Neural Network**: A deep learning model built with `TensorFlow`/`Keras`.
    - **Random Forest Regressor**: An ensemble learning model from `scikit-learn`.
- **Evaluation**: Calculates Mean Squared Error (MSE), Root Mean Square Error (RMSE), and R² Score for each model.
- **Visualization**: Generates plots of actual vs. predicted values for both test and out-of-sample evaluation periods.
- **Modular Structure**: Code is organized into separate modules for data collection, preprocessing, modeling, evaluation, and a main execution script.
- **Dynamic Output**: All output filenames, plot titles, and report content dynamically reflect the chosen ticker symbol and date ranges.

## Project Structure

```
stock_prediction/
├── data/                     # Stores downloaded CSV data
├── plots/                    # Stores generated prediction plots
├── reports/                  # Stores performance reports
├── data_collector.py         # Handles data downloading from Yahoo Finance
├── preprocessor.py           # Manages data cleaning, feature engineering, and splitting
├── linear_regression_model.py # Implements the Linear Regression model
├── lstm_model.py             # Implements the LSTM Neural Network model
├── random_forest_model.py    # Implements the Random Forest Regressor model
├── evaluator.py              # Handles model evaluation and plot generation
├── main.py                   # Main script to run the entire pipeline
└── requirements.txt          # Lists all necessary Python packages
```

## Setup and Installation

1.  **Clone the repository (or create the files manually):**

    ```bash
    git clone <repository_url>
    cd stock_prediction
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python3.11 -m venv venv
    source venv/bin/activate
    ```

3.  **Install required libraries:**

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

All primary configurations are managed in `main.py`:

-   `TICKER_SYMBOL`: The stock ticker you want to analyze (e.g., `AAPL`, `^GSPC`).
-   `TRAIN_START_DATE`, `TRAIN_END_DATE`: Date range for training and initial testing.
-   `EVAL_START_DATE`, `EVAL_END_DATE`: Date range for out-of-sample evaluation.
-   `LAG_STEPS`: Number of past days to use for lag features.
-   `TEST_SIZE`: Proportion of data for the test set (from the training period).
-   `LSTM_EPOCHS`, `LSTM_BATCH_SIZE`: Hyperparameters for the LSTM model.
-   `RF_N_ESTIMATORS`, `RF_MAX_DEPTH`: Hyperparameters for the Random Forest model.

## How to Run

Execute the `main.py` script from the `stock_prediction` directory:

```bash
python3.11 main.py
```

The script will perform the following steps:

1.  Download historical stock data.
2.  Preprocess the data, create features (lagged daily returns, moving averages), and define the target (next day's daily return).
3.  Train Linear Regression, Random Forest, and LSTM models.
4.  Make predictions on both the in-sample test set and the out-of-sample evaluation set.
5.  Evaluate model performance using MSE, RMSE, and R² Score.
6.  Generate and save plots of actual vs. predicted prices.
7.  Generate and save a combined performance report.

## Academic Relevance (IB Computer Science Extended Essay)

This project provides a practical application of machine learning concepts to a real-world problem (stock price prediction). It demonstrates:

-   **Data Handling**: Proficiency in collecting, cleaning, and transforming raw data.
-   **Algorithm Implementation**: Understanding and implementing different machine learning algorithms (regression, neural networks, ensemble methods).
-   **Model Evaluation**: Critical analysis of model performance using appropriate metrics and visualization techniques.
-   **Modular Design**: Adherence to good programming practices through a well-structured and documented codebase.
-   **Time-Series Analysis**: Specific considerations for time-series data, such as chronological splitting and feature engineering (lag features, moving averages).
-   **Addressing Challenges**: The iterative process of identifying and resolving issues like data leakage and model underperformance, which is crucial in real-world machine learning projects.

You can extend this project by:

-   Experimenting with more advanced feature engineering (e.g., technical indicators like RSI, MACD).
-   Exploring other machine learning models (e.g., ARIMA, Prophet, GRU).
-   Implementing hyperparameter tuning techniques (e.g., GridSearchCV, RandomizedSearchCV).
-   Analyzing different stock tickers or market indices.
-   Investigating the impact of different time windows or data frequencies.
-   Incorporating sentiment analysis from news data.

## License

This project is open-source

