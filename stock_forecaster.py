#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for stock price forecasting system with sentiment analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

from src.data_loader import StockDataLoader
from src.forecasting import Forecaster
from src.sentiment_analysis import SentimentAnalyzer
from src.utils import calculate_metrics, plot_forecasts
from src.combined_model import CombinedModel

def main():
    # Configuration
    ticker = "D05.SI"  # UOB Singapore
    start_date = "2018-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    forecast_horizon = 30  # Days to forecast
    training_window = 30  # Days of training data to use
    
    # Validate configuration
    if forecast_horizon <= 0 or training_window <= 0:
        raise ValueError("Forecast horizon and training window must be positive")
    if forecast_horizon > training_window:
        print("Warning: Forecast horizon is larger than training window")
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Load stock data
    data_loader = StockDataLoader()
    stock_data = data_loader.load_stock_data(ticker, start_date, end_date)
    
    # Validate stock data
    if len(stock_data) == 0:
        raise ValueError("Stock data is empty")
    print(f"\nStock data range: {stock_data.start_time()} to {stock_data.end_time()}")
    print(f"Number of data points: {len(stock_data)}")
    
    # Check for missing values in stock data
    stock_values = stock_data.values()
    if np.any(np.isnan(stock_values)):
        print("Warning: NaN values detected in Stock data")
        stock_data = stock_data.fillna(method='ffill')
    
    # Check for outliers in stock data
    mean_price = np.mean(stock_values)
    std_price = np.std(stock_values)
    outliers = np.abs(stock_values - mean_price) > 3 * std_price
    if np.any(outliers):
        print(f"Warning: {np.sum(outliers)} potential outliers detected in stock data")
    
    # Split into train/test
    train_data, test_data = data_loader.split_data(stock_data, test_size=forecast_horizon)

    # Validate train/test split
    if len(train_data) == 0 or len(test_data) == 0:
        raise ValueError("Train or test data is empty after split")
    print(f"Training data range: {train_data.start_time()} to {train_data.end_time()}")
    print(f"Test data range: {test_data.start_time()} to {test_data.end_time()}")
    
    # Check for sufficient training data
    min_required_points = max(training_window, forecast_horizon * 2)
    if len(train_data) < min_required_points:
        raise ValueError(f"Insufficient training data. Need at least {min_required_points} points, got {len(train_data)}")

    # Get the date range from the stock data
    stock_start_date = stock_data.start_time()
    stock_end_date = stock_data.end_time()

    # Sentiment analysis with matching date range
    sentiment_analyzer = SentimentAnalyzer()
    # We can only extract sentiment from now till 1 month ago
    sentiment_data = sentiment_analyzer.analyze_sentiment(
        ticker,
        #start_date=stock_start_date,
        #end_date=stock_end_date 
    )
    
    # Validate sentiment data
    if len(sentiment_data) == 0:
        raise ValueError("Sentiment data is empty")
    print(f"\nSentiment data range: {sentiment_data.start_time()} to {sentiment_data.end_time()}")
    print(f"Number of sentiment data points: {len(sentiment_data)}")
    
    # Check for data gaps in sentiment
    sentiment_dates = pd.date_range(start=sentiment_data.start_time(), end=sentiment_data.end_time(), freq='D')
    if len(sentiment_dates) != len(sentiment_data):
        print("Warning: Gaps detected in sentiment data")
        # Check for large gaps
        sentiment_index = pd.DatetimeIndex(sentiment_data.time_index)
        gaps = sentiment_index.to_series().diff().dt.days > 1
        if gaps.any():
            print(f"Large gaps detected in sentiment data: {gaps[gaps].index.tolist()}")
    
    # Check sentiment data quality
    sentiment_values = sentiment_data.values()
    if np.any(np.isnan(sentiment_values)):
        print("Warning: NaN values detected in sentiment data")
        sentiment_data = sentiment_data.fillna(method='ffill')
    
    fear_greed_data = sentiment_analyzer.get_fear_greed_index(
        start_date=stock_start_date,
        end_date=stock_end_date
    )
    
    # Validate fear/greed data
    if len(fear_greed_data) == 0:
        raise ValueError("Fear/Greed data is empty")
    print(f"\nFear/Greed data range: {fear_greed_data.start_time()} to {fear_greed_data.end_time()}")
    print(f"Number of fear/greed data points: {len(fear_greed_data)}")
    
    # Check for data gaps in fear/greed
    fear_greed_dates = pd.date_range(start=fear_greed_data.start_time(), end=fear_greed_data.end_time(), freq='D')
    if len(fear_greed_dates) != len(fear_greed_data):
        print("Warning: Gaps detected in fear/greed data")
        # Check for large gaps
        fear_greed_index = pd.DatetimeIndex(fear_greed_data.time_index)
        gaps = fear_greed_index.to_series().diff().dt.days > 1
        if gaps.any():
            print(f"Large gaps detected in fear/greed data: {gaps[gaps].index.tolist()}")
    
    # Check fear/greed data quality
    fear_greed_values = fear_greed_data.values()
    if np.any(np.isnan(fear_greed_values)):
        print("Warning: NaN values detected in fear/greed data")
        fear_greed_data = fear_greed_data.fillna(method='ffill')
    
    # Check for value range in fear/greed data
    if np.any(fear_greed_values < 0) or np.any(fear_greed_values > 100):
        print("Warning: Fear/Greed values outside expected range [0, 100]")
    
    # Initialize forecaster with various models
    forecaster = Forecaster()
    
    # Add models to the forecaster
    forecaster.add_benchmark_models()
    forecaster.add_statistical_models()
    forecaster.add_machine_learning_models()
    
    # Train and evaluate models
    results, trained_models = forecaster.evaluate_all(train_data, test_data, forecast_horizon)
    
    # Validate evaluation results
    if not results:
        raise ValueError("No evaluation results returned")
    print("\nForecasting Results (RMSE):")
    for model_name, error in results.items():
        if error <= 0:
            print(f"Warning: Invalid RMSE value for {model_name}: {error}")
        print(f"{model_name}: {error:.4f}")
    
    # Find benchmark models and their average RMSE
    benchmark_models = ['naive', 'seasonal_naive', 'drift']
    benchmark_rmse = np.mean([results[model] for model in benchmark_models if model in results])
    print(f"Average benchmark RMSE: {benchmark_rmse:.4f}")
    
    # Select models that outperform benchmarks
    outperforming_models = {
        name: error for name, error in results.items() 
        if error < benchmark_rmse and name not in benchmark_models
    }
    
    if not outperforming_models:
        print("Warning: No models outperformed the benchmarks. Using all non-benchmark models.")
        outperforming_models = {
            name: error for name, error in results.items() 
            if name not in benchmark_models
        }
    
    print("\nModels selected for ensemble:")
    for model_name, rmse in outperforming_models.items():
        print(f"{model_name}: {rmse:.4f}")
    
    # Combine train and test data for final training
    full_data = train_data.concatenate(test_data)
    print(f"\nRetraining models with full data from {full_data.start_time()} to {full_data.end_time()}")
    
    # Generate predictions for each selected model using full data
    model_predictions = {}
    for model_name in outperforming_models.keys():
        print(f"\nRetraining {model_name} with full data...")
        if model_name in trained_models:
            # Get a fresh instance of the model
            model = forecaster.models[model_name]
            # Retrain with full data
            model.fit(full_data)
            # Generate predictions
            model_predictions[model_name] = model.predict(forecast_horizon)
            print(f"Generated {forecast_horizon}-day forecast for {model_name}")
        else:
            print(f"Warning: Trained model not found for {model_name}")
    
    # Combined model - direct prediction without evaluation
    print("\nGenerating 30-day forecast using combined model...")
    
    # Find the common date range across all datasets
    common_start = max(
        stock_data.start_time(),
        sentiment_data.start_time(),
        fear_greed_data.start_time()
    )
    common_end = min(
        stock_data.end_time(),
        sentiment_data.end_time(),
        fear_greed_data.end_time()
    )
    
    print(f"\nUsing data from {common_start} to {common_end}")
    
    # Slice all data to the common date range
    training_data = stock_data.slice(common_start, common_end)
    training_sentiment = sentiment_data.slice(common_start, common_end)
    training_fear_greed = fear_greed_data.slice(common_start, common_end)
    
    # Validate data
    if len(training_data) == 0:
        raise ValueError("Stock data is empty after alignment")
    if len(training_sentiment) == 0:
        raise ValueError("Sentiment data is empty after alignment")
    if len(training_fear_greed) == 0:
        raise ValueError("Fear/Greed data is empty after alignment")
    
    # Ensure we have enough data points
    # min_required_points = 30  # Minimum number of points needed for training
    # if len(training_data) < min_required_points:
    #     raise ValueError(f"Insufficient data points. Need at least {min_required_points}, got {len(training_data)}")
    
    print(f"Training data points: {len(training_data)}")
    print(f"Training sentiment points: {len(training_sentiment)}")
    print(f"Training fear/greed points: {len(training_fear_greed)}")
    
    # Check for data alignment
    if not all(len(ts) == len(training_data) for ts in [training_sentiment, training_fear_greed]):
        print("Warning: Data lengths are not aligned")
        print(f"Stock data length: {len(training_data)}")
        print(f"Sentiment data length: {len(training_sentiment)}")
        print(f"Fear/Greed data length: {len(training_fear_greed)}")
        
        # Resample all data to daily frequency to ensure alignment
        print("Resampling data to daily frequency...")
        training_data = training_data.resample(freq='D')
        training_sentiment = training_sentiment.resample(freq='D')
        training_fear_greed = training_fear_greed.resample(freq='D')
        
        # Fill any missing values after resampling
        training_data = training_data.fillna(method='ffill')
        training_sentiment = training_sentiment.fillna(method='ffill')
        training_fear_greed = training_fear_greed.fillna(method='ffill')
    
    # Check for data continuity
    for name, data in [("Stock", training_data), ("Sentiment", training_sentiment), ("Fear/Greed", training_fear_greed)]:
        dates = pd.DatetimeIndex(data.time_index)
        if not dates.is_monotonic_increasing:
            print(f"Warning: {name} data dates are not monotonically increasing")
            # Sort the data by date
            data = data.sort_index()
        gaps = dates.to_series().diff().dt.days > 1
        if gaps.any():
            print(f"Warning: Gaps detected in {name} data: {gaps[gaps].index.tolist()}")
    
    # Final validation of data
    if len(training_data) == 0 or len(training_sentiment) == 0 or len(training_fear_greed) == 0:
        raise ValueError("Data is empty after processing")
    
    if not all(len(ts) == len(training_data) for ts in [training_sentiment, training_fear_greed]):
        raise ValueError("Data lengths are still not aligned after processing")
    
    # Train the model on all available data
    print("\nTraining combined model...")
    combined_model = CombinedModel(model_type="gradient_boosting")
    combined_model.train(training_data, training_sentiment, training_fear_greed)
    
    # Generate prediction for the next forecast_horizon days
    print("Generating predictions...")
    combined_prediction = combined_model.predict(forecast_horizon)
    
    # Validate prediction
    if len(combined_prediction) == 0:
        raise ValueError("Generated prediction is empty")
    if len(combined_prediction) != forecast_horizon:
        raise ValueError(f"Prediction length {len(combined_prediction)} does not match forecast horizon {forecast_horizon}")
    
    # Check prediction values
    pred_values = combined_prediction.values()
    if np.any(np.isnan(pred_values)):
        raise ValueError("NaN values detected in predictions")
    if np.any(np.isinf(pred_values)):
        raise ValueError("Infinite values detected in predictions")
    
    # Check for unrealistic predictions
    last_price = training_data.values()[-1]
    max_change = 0.2  # 20% maximum allowed change
    if np.any(np.abs(pred_values - last_price) / last_price > max_change):
        print("Warning: Large price changes detected in predictions")
    
    print(f"\nPrediction range: {combined_prediction.start_time()} to {combined_prediction.end_time()}")
    print(f"Number of predicted points: {len(combined_prediction)}")
    print(f"Prediction value range: {np.min(pred_values):.2f} to {np.max(pred_values):.2f}")
    
    # Calculate weights using inverse RMSE (better performance = higher weight)
    # Add combined model with a weight based on its historical performance
    # For the combined model, we'll use the average RMSE of the best performing models
    combined_model_weight = 1 / np.mean(list(outperforming_models.values()))
    weights = {name: 1/rmse for name, rmse in outperforming_models.items()}
    weights['combined'] = combined_model_weight
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    weights = {name: weight/total_weight for name, weight in weights.items()}
    
    print("\nModel weights in ensemble:")
    for model_name, weight in weights.items():
        print(f"{model_name}: {weight:.4f}")
    
    # Create weighted ensemble prediction
    ensemble_prediction = np.zeros(forecast_horizon)
    all_predictions = []  # Store all model predictions for confidence interval calculation
    
    # First, collect all predictions
    for model_name, prediction in model_predictions.items():
        all_predictions.append(prediction.values().flatten())
    all_predictions.append(combined_prediction.values().flatten())
    
    # Convert to numpy array for easier calculation
    all_predictions = np.array(all_predictions)
    
    # Calculate weighted mean and standard deviation
    weights_array = np.array(list(weights.values()))
    weighted_mean = np.average(all_predictions, weights=weights_array, axis=0)
    weighted_std = np.sqrt(np.average((all_predictions - weighted_mean)**2, weights=weights_array, axis=0))
    
    # Calculate 95% confidence intervals
    z_score = 1.96  # 95% confidence interval
    upper_bound = weighted_mean + z_score * weighted_std
    lower_bound = weighted_mean - z_score * weighted_std
    
    # Save the ensemble prediction with confidence intervals
    prediction_df = pd.DataFrame({
        'Date': combined_prediction.time_index,
        'Predicted_Price': weighted_mean,
        'Upper_Bound': upper_bound,
        'Lower_Bound': lower_bound
    })
    prediction_df.to_csv(f"results/{ticker.replace('.', '_')}_30day_forecast.csv", index=False)
    
    print(f"\n30-day forecast saved to results/{ticker.replace('.', '_')}_30day_forecast.csv")

    # Plot historical and predicted prices
    plt.figure(figsize=(15, 8))
    
    # Plot historical data
    historical_df = pd.DataFrame({
        'Date': stock_data.time_index,
        'Price': stock_data.values().flatten()
    })
    plt.plot(historical_df['Date'], historical_df['Price'], label='Historical Price', color='blue', linewidth=2)
    
    # Plot individual model predictions
    colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i, (model_name, prediction) in enumerate(model_predictions.items()):
        color = colors[i % len(colors)]
        plt.plot(prediction.time_index, prediction.values().flatten(), 
                label=f'{model_name} Prediction', 
                color=color, 
                linestyle='--', 
                alpha=0.5)
    
    # Plot Sentiment Analysis Model prediction
    plt.plot(combined_prediction.time_index, combined_prediction.values().flatten(),
            label='Sentiment Analysis Model',
            color='magenta',
            linewidth=2,
            linestyle='-.')
    
    # Plot ensemble prediction with confidence intervals
    plt.plot(prediction_df['Date'], prediction_df['Predicted_Price'], 
            label='Ensemble Prediction', 
            color='black', 
            linewidth=2, 
            linestyle='-')
    
    # Plot confidence intervals
    plt.fill_between(prediction_df['Date'], 
                    prediction_df['Lower_Bound'], 
                    prediction_df['Upper_Bound'], 
                    color='gray', 
                    alpha=0.2, 
                    label='95% Confidence Interval')
    
    # Customize the plot
    plt.title(f'{ticker} Stock Price Forecast (Weighted Ensemble with 95% CI)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside
    plt.grid(True)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"results/{ticker.replace('.', '_')}_forecast_plot.png", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main() 