#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for loading and preprocessing stock data.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.utils import missing_values
import os
from datetime import datetime, timedelta

class StockDataLoader:
    """Handles loading and preprocessing of stock data for forecasting."""
    
    def __init__(self):
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def load_stock_data(self, ticker, start_date, end_date, force_download=False):
        """
        Load stock data for the specified ticker, either from file or Yahoo Finance.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            force_download (bool): Force download even if file exists
            
        Returns:
            TimeSeries: Darts TimeSeries object with the closing prices
        """
        file_path = os.path.join(self.data_dir, f"{ticker.replace('.', '_')}.csv")
        
        if os.path.exists(file_path) and not force_download:
            print(f"Loading {ticker} data from file")
            df = pd.read_csv(file_path, index_col=0)
            df = df[3:].reset_index().rename(columns={'Price': 'Date'})
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        else:
            print(f"Downloading {ticker} data from Yahoo Finance")
            df = yf.download(ticker, start=start_date, end=end_date)
            df.to_csv(file_path)
        
        # Create Darts TimeSeries from the closing prices
        ts = TimeSeries.from_dataframe(df, value_cols=['Close'], time_col='Date', fill_missing_dates=True, freq='D')
        ts = missing_values.fill_missing_values(ts, fill="auto", method="ffill", limit_direction="forward")
        return ts
    
    def split_data(self, time_series, test_size=30, val_size=None):
        """
        Split the time series data into training and test sets.
        
        Args:
            time_series (TimeSeries): Input time series
            test_size (int): Number of days for testing
            val_size (int, optional): Number of days for validation
            
        Returns:
            tuple: (train_data, test_data) or (train_data, val_data, test_data)
        """
        if val_size:
            train, val, test = time_series[:-val_size-test_size], time_series[-val_size-test_size:-test_size], time_series[-test_size:]
            return train, val, test
        else:
            train, test = time_series[:-test_size], time_series[-test_size:]
            return train, test
    
    def scale_data(self, train_data, test_data, val_data=None):
        """
        Scale the data using Darts Scaler.
        
        Args:
            train_data (TimeSeries): Training data
            test_data (TimeSeries): Test data
            val_data (TimeSeries, optional): Validation data
            
        Returns:
            tuple: Scaled data in the same format as inputs
        """
        scaler = Scaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)
        
        if val_data is not None:
            val_scaled = scaler.transform(val_data)
            return train_scaled, val_scaled, test_scaled, scaler
        
        return train_scaled, test_scaled, scaler
    
    def load_multiple_tickers(self, tickers, start_date, end_date, force_download=False):
        """
        Load multiple tickers into a dictionary of TimeSeries.
        
        Args:
            tickers (list): List of ticker symbols
            start_date (str): Start date
            end_date (str): End date
            force_download (bool): Force download
            
        Returns:
            dict: Dictionary of ticker to TimeSeries
        """
        data = {}
        for ticker in tickers:
            data[ticker] = self.load_stock_data(ticker, start_date, end_date, force_download)
        return data 