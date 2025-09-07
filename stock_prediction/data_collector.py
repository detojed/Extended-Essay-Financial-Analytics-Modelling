
import yfinance as yf
import pandas as pd

def download_data(ticker: str, start_date: str, end_date: str, output_path: str):
    """
    Downloads historical stock data from Yahoo Finance and saves it to a CSV file.

    Args:
        ticker (str): The stock ticker symbol (e.g., 
'^GSPC' for S&P 500).
        start_date (str): Start date for data download in 
'YYYY-MM-DD' format.
        end_date (str): End date for data download in 
'YYYY-MM-DD' format.
        output_path (str): Path to save the downloaded data as a CSV file.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if not data.empty:
            # Flatten the multi-level column headers
            data.columns = ['_'.join(col).strip() for col in data.columns.values]
            data.to_csv(output_path)
            print(f"Successfully downloaded data for {ticker} and saved to {output_path}")
        else:
            print(f"No data downloaded for {ticker} between {start_date} and {end_date}")
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")




