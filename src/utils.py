#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for the stock price forecasting system.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.metrics import rmse
import os
from datetime import datetime

def calculate_metrics(actual, predicted):
    """
    Calculate error metrics between actual and predicted values.
    
    Args:
        actual (TimeSeries): Actual values
        predicted (TimeSeries): Predicted values
        
    Returns:
        dict: Dictionary of error metrics
    """
    metrics = {
        'rmse': rmse(actual, predicted)
    }
    return metrics

def plot_forecasts(historical_data, actual_data, forecasts, model_names=None):
    """
    Plot multiple forecasts against actual data.
    
    Args:
        historical_data (TimeSeries): Historical data
        actual_data (TimeSeries): Actual future data
        forecasts (list): List of forecast TimeSeries
        model_names (list, optional): List of model names
    """
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(forecasts))]
    
    plt.figure(figsize=(15, 8))
    
    # Plot historical and actual data
    historical_data.plot(label='Historical')
    actual_data.plot(label='Actual')
    
    # Plot forecasts
    for forecast, model_name in zip(forecasts, model_names):
        forecast.plot(label=model_name)
    
    plt.title('Stock Price Forecasts vs Actual')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/forecasts_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.show()

def compare_models(actual_data, forecasts, model_names=None):
    """
    Compare multiple models based on their forecasts.
    
    Args:
        actual_data (TimeSeries): Actual data
        forecasts (list): List of forecast TimeSeries
        model_names (list, optional): List of model names
        
    Returns:
        DataFrame: DataFrame with model comparison metrics
    """
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(forecasts))]
    
    results = []
    for forecast, model_name in zip(forecasts, model_names):
        metrics = calculate_metrics(actual_data, forecast)
        metrics['model'] = model_name
        results.append(metrics)
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['model'] + [col for col in results_df.columns if col != 'model']
    results_df = results_df[cols]
    
    # Sort by RMSE
    results_df = results_df.sort_values('rmse')
    
    return results_df

def plot_model_comparison(results_df, metric='rmse'):
    """
    Plot model comparison based on RMSE metric.
    
    Args:
        results_df (DataFrame): DataFrame with model comparison results
        metric (str): Metric to plot (only 'rmse' is supported)
    """
    plt.figure(figsize=(12, 6))
    
    # Create horizontal bar chart
    plt.barh(results_df['model'], results_df[metric])
    
    plt.title(f'Model Comparison - {metric.upper()}')
    plt.xlabel(metric.upper())
    plt.ylabel('Model')
    plt.grid(True, axis='x')
    
    # Add values to bars
    for i, value in enumerate(results_df[metric]):
        plt.text(value, i, f' {value:.4f}', va='center')
    
    plt.tight_layout()
    plt.savefig(f'results/model_comparison_{metric}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.show()

def create_html_report(ticker, results_df, plot_paths=None):
    """
    Create an HTML report with model results and plots.
    
    Args:
        ticker (str): Ticker symbol
        results_df (DataFrame): DataFrame with model comparison results
        plot_paths (list, optional): List of paths to plots to include
        
    Returns:
        str: Path to the HTML report
    """
    if plot_paths is None:
        plot_paths = []
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Create report filename
    report_path = f"results/{ticker.replace('.', '_')}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    
    # Create HTML content
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stock Price Forecasting Report - {ticker}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
            th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #2c3e50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .plot {{ margin: 20px 0; text-align: center; }}
            .plot img {{ max-width: 100%; }}
        </style>
    </head>
    <body>
        <h1>Stock Price Forecasting Report</h1>
        <p><strong>Ticker:</strong> {ticker}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Model Comparison</h2>
        <table>
            <tr>
                {' '.join(f'<th>{col}</th>' for col in results_df.columns)}
            </tr>
            {' '.join(f'<tr>{" ".join(f"<td>{cell}</td>" for cell in row)}</tr>' for row in results_df.values.tolist())}
        </table>
        
        <h2>Plots</h2>
        {' '.join(f'<div class="plot"><img src="../{path}" alt="Plot"></div>' for path in plot_paths)}
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(report_path, 'w') as f:
        f.write(html)
    
    return report_path

def analyze_seasonality(time_series, m=None):
    """
    Analyze seasonality in the time series.
    
    Args:
        time_series (TimeSeries): Time series to analyze
        m (int, optional): Period of the seasonality to check. If None, will check common values.
        
    Returns:
        dict: Dictionary with seasonality analysis results
    """
    from darts.utils.statistics import check_seasonality, extract_trend_and_seasonality
    
    # Convert to pandas Series for analysis
    series = time_series.pd_series().iloc[:, 0]
    
    results = {}
    
    # Check common seasonality periods if m is not provided
    if m is None:
        for period in [5, 7, 14, 30, 90, 365]:
            if len(series) >= period * 2:  # Need at least 2 periods of data
                is_seasonal, p_value = check_seasonality(time_series, m=period)
                results[period] = {
                    'is_seasonal': is_seasonal,
                    'p_value': p_value
                }
    else:
        is_seasonal, p_value = check_seasonality(time_series, m=m)
        results[m] = {
            'is_seasonal': is_seasonal,
            'p_value': p_value
        }
    
    # Extract trend and seasonality components
    if len(series) >= 14:  # Need sufficient data
        try:
            trend, seasonal = extract_trend_and_seasonality(time_series, m=7)
            results['decomposition'] = {
                'trend': trend,
                'seasonal': seasonal
            }
        except Exception as e:
            results['decomposition_error'] = str(e)
    
    return results

def plot_seasonality_components(time_series, results):
    """
    Plot the original time series along with its trend and seasonality components.
    
    Args:
        time_series (TimeSeries): Original time series
        results (dict): Results from analyze_seasonality
    """
    if 'decomposition' not in results:
        print("Decomposition results not available")
        return
    
    trend = results['decomposition']['trend']
    seasonal = results['decomposition']['seasonal']
    
    plt.figure(figsize=(15, 10))
    
    # Plot original series
    plt.subplot(3, 1, 1)
    time_series.plot(label='Original')
    plt.title('Original Time Series')
    plt.grid(True)
    plt.legend()
    
    # Plot trend
    plt.subplot(3, 1, 2)
    trend.plot(label='Trend')
    plt.title('Trend Component')
    plt.grid(True)
    plt.legend()
    
    # Plot seasonality
    plt.subplot(3, 1, 3)
    seasonal.plot(label='Seasonality')
    plt.title('Seasonal Component')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/seasonality_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.show() 