#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for time series forecasting with various models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts.models import (
    NaiveSeasonal,
    NaiveDrift, 
    Prophet,
    ARIMA,
    AutoCES,
    AutoETS,
    AutoMFLES,
    AutoTBATS,
    AutoTheta,
    AutoARIMA,
    ExponentialSmoothing,
    Theta,
    RegressionModel,
    FFT,
    KalmanForecaster,
    RNNModel,
    TCNModel,
    TransformerModel,
    TFTModel,
    NBEATSModel,
    BlockRNNModel,
    XGBModel
)
from darts.metrics import rmse
from darts.utils.statistics import check_seasonality, extract_trend_and_seasonality
from darts.dataprocessing.transformers import Scaler
from pytorch_lightning.callbacks import EarlyStopping
import os
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
import torch
from typing import Tuple, List, Dict
from darts.timeseries import TimeSeries

# GPU Configuration
def setup_gpu():
    """
    Configure GPU settings and verify CUDA availability.
    """
    if torch.cuda.is_available():
        print(f"GPU is available. Using device: {torch.cuda.get_device_name(0)}")
        # Set default tensor type to cuda        # Set device
        device = torch.device("cuda")
        # Enable cuDNN benchmarking for better performance
        torch.backends.cudnn.benchmark = True
        return device
    else:
        print("GPU is not available. Using CPU instead.")
        device = torch.device("cpu")
        return device

# Initialize GPU
DEVICE = setup_gpu()

class Forecaster:
    """Handles forecasting using various models."""
    
    def __init__(self):
        self.models = {}
        self.forecasts = {}
        self.results = {}
        self.test_results = {}  # Store test metrics
        self.scaler = None
        self.device = DEVICE
        self.trained_models = {}  # Store trained models
        
    def add_benchmark_models(self):
        """Add benchmark models."""
        self.models['naive'] = NaiveSeasonal(K=1)
        self.models['seasonal_naive'] = NaiveSeasonal(K=5)  # Weekly seasonality
        self.models['drift'] = NaiveDrift()
        
    def add_statistical_models(self):
        """Add statistical models."""
        self.models['arima'] = AutoARIMA()
        self.models['complex_exponential_smoothing'] = AutoCES()
        self.models['ets'] = AutoETS()
        self.models['mfles'] = AutoMFLES(test_size=0.2)
        self.models['tbats'] = AutoTBATS(season_length=7)
        self.models['theta'] = AutoTheta()
        self.models['exponential_smoothing'] = ExponentialSmoothing(
            seasonal_periods=5  # Weekly seasonality
        )
        self.models['prophet'] = Prophet()
        self.models['fft'] = FFT(nr_freqs_to_keep=14)
        
    def add_machine_learning_models(self):
        """Add enhanced machine learning models with optimized hyperparameters."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
        
        # Enhanced linear model
        self.models['linear'] = RegressionModel(
            model=Ridge(alpha=1.0),  # Adjusted alpha for regularization
            lags=14,
            output_chunk_length=1
        )
        
        # Enhanced Random Forest regressor
        self.models['random_forest'] = RegressionModel(
            model=RandomForestRegressor(
                n_estimators=200,  # Increased number of trees
                max_depth=15,      # Increased depth for more complexity
                min_samples_split=4,
                random_state=42
            ),
            lags=14,
            output_chunk_length=1
        )
        
        # Simplified RNN Model
        self.models['rnn'] = RNNModel(
            input_chunk_length=14,           # Reduced for smaller datasets
            output_chunk_length=1,
            model="GRU",                     # Using GRU for potentially better performance
            hidden_dim=20,                   # Reduced complexity
            n_rnn_layers=1,                  # Single layer
            dropout=0.1,                     # Reduced dropout
            batch_size=16,                   # Smaller batch size
            n_epochs=50,                     # Fewer epochs
            optimizer_kwargs={"lr": 1e-3},   
            loss_fn=torch.nn.MSELoss(),      # Explicit loss function
            random_state=42,
            force_reset=True,
            save_checkpoints=True,           # Save checkpoints
            work_dir="./models/rnn_checkpoints"
        )
        
        # Simplified Transformer Model
        self.models['transformer'] = TransformerModel(
            input_chunk_length=14,           # Reduced for smaller datasets
            output_chunk_length=1,
            d_model=32,                      # Reduced complexity
            nhead=4,
            num_encoder_layers=1,            # Single layer
            num_decoder_layers=1,            # Single layer
            dim_feedforward=64,              # Reduced feedforward network size
            dropout=0.1,                     # Reduced dropout
            batch_size=16,                   # Smaller batch size
            n_epochs=50,                     # Fewer epochs
            optimizer_kwargs={"lr": 1e-3},
            loss_fn=torch.nn.MSELoss(),      # Explicit loss function
            random_state=42,
            force_reset=True,
            save_checkpoints=True,           # Save checkpoints
            work_dir="./models/transformer_checkpoints"
        )
        
        # Simplified TCN Model
        self.models['tcn'] = TCNModel(
            input_chunk_length=14,           # Reduced for smaller datasets
            output_chunk_length=1,
            kernel_size=3,
            num_filters=16,                  # Keep filters
            dilation_base=2,
            dropout=0.1,                     # Reduced dropout
            batch_size=16,                   # Smaller batch size
            n_epochs=50,                     # Fewer epochs
            optimizer_kwargs={"lr": 1e-3},
            loss_fn=torch.nn.MSELoss(),      # Explicit loss function
            random_state=42,
            force_reset=True,
            save_checkpoints=True,           # Save checkpoints
            work_dir="./models/tcn_checkpoints"
        )
        
        # Simplified N-BEATS Model
        self.models['nbeats'] = NBEATSModel(
            input_chunk_length=14,           # Reduced for smaller datasets
            output_chunk_length=1,
            generic_architecture=True,       # Use generic architecture
            num_stacks=2,                    # Reduced stacks
            num_blocks=2,                    # Reduced blocks
            num_layers=2,                    # Reduced layers
            layer_widths=32,                 # Reduced layer width
            batch_size=16,                   # Smaller batch size
            n_epochs=50,                     # Fewer epochs
            optimizer_kwargs={"lr": 1e-3},
            loss_fn=torch.nn.MSELoss(),      # Explicit loss function
            random_state=42,
            force_reset=True,
            save_checkpoints=True,           # Save checkpoints
            work_dir="./models/nbeats_checkpoints"
        )

        # Simplified XGBoost Model
        self.models['xgboost'] = XGBModel(
            lags=14,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
    
    def blocked_cross_validation(self, 
                               time_series: pd.Series,
                               n_splits: int = 5,
                               train_size: int = None,
                               validation_size: int = None,
                               gap: int = 5) -> List[Tuple[pd.Series, pd.Series, pd.Series]]:
        """
        Perform blocked cross-validation for time series data.
        
        Args:
            time_series (pd.Series): Input time series data
            n_splits (int): Number of splits for cross-validation
            train_size (int): Size of training set for each fold
            validation_size (int): Size of validation set for each fold
            gap (int): Gap size between train and validation sets
            
        Returns:
            List of tuples containing (train, validation, test) sets for each fold
        """
        n_samples = len(time_series)
        
        if train_size is None:
            train_size = n_samples // (n_splits + 2)
        if validation_size is None:
            validation_size = n_samples // (n_splits + 2)
            
        folds = []
        for i in range(n_splits):
            # Calculate indices for this fold
            train_end = n_samples - (n_splits - i) * (train_size + validation_size + gap)
            val_end = train_end + train_size
            test_end = val_end + gap + validation_size
            
            if test_end > n_samples:
                break
                
            # Split the data
            train_data = time_series.iloc[train_end:val_end]
            val_data = time_series.iloc[val_end + gap:test_end]
            test_data = time_series.iloc[test_end:test_end + validation_size]
            
            folds.append((train_data, val_data, test_data))
            
        return folds

    def train_with_validation(self, 
                            model_name: str,
                            train_data: pd.Series,
                            val_data: pd.Series,
                            early_stopping_patience: int = 5) -> None:
        """
        Train a model with validation set and early stopping.
        
        Args:
            model_name (str): Name of the model to train
            train_data (pd.Series): Training data
            val_data (pd.Series): Validation data
            early_stopping_patience (int): Number of epochs to wait before early stopping
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        # Check for zero or negative values
        train_values = train_data.values()
        val_values = val_data.values()
        
        if np.any(train_values <= 0) or np.any(val_values <= 0):
            if model_name in ['fft', 'tbats', 'mfles']:
                print(f"\nWARNING: Zero or negative values detected. {model_name} may not perform well.")
                print("Consider using adjusted prices or checking data quality.")
                
                # Add small epsilon to zero values for these models
                epsilon = 1e-8
                if np.any(train_values == 0):
                    train_values[train_values == 0] = epsilon
                    train_data = TimeSeries.from_times_and_values(
                        train_data.time_index,
                        train_values
                    )
                if np.any(val_values == 0):
                    val_values[val_values == 0] = epsilon
                    val_data = TimeSeries.from_times_and_values(
                        val_data.time_index,
                        val_values
                    )
            
        model = self.models[model_name]
        
        # Add validation callback for deep learning models
        if isinstance(model, (RNNModel, TCNModel, TransformerModel, NBEATSModel)):
            print(f"Training {model_name} on {self.device}")
            model.fit(
                train_data,
                val_series=val_data,
                early_stopping=True,
                early_stopping_patience=early_stopping_patience,
                verbose=True
            )
        else:
            try:
                model.fit(train_data)
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                self.test_results[model_name] = float('inf')
                return
            
        # Calculate test metrics
        try:
            test_forecast = model.predict(len(val_data))
            
            # Verify test forecast data
            if test_forecast is None or len(test_forecast) == 0:
                print(f"Warning: {model_name} produced empty forecast")
                self.test_results[model_name] = float('inf')  # Use infinity as a fallback
                return
            
            # Check for NaN values in forecast
            if np.any(np.isnan(test_forecast.values())):
                print(f"Warning: {model_name} produced NaN values in forecast")
                # Fill NaN values with the last valid value
                test_forecast = test_forecast.fillna(method='ffill').fillna(method='bfill')
                
            # Ensure both series have the same time index
            if not np.array_equal(test_forecast.time_index, val_data.time_index):
                print(f"Warning: Time index mismatch for {model_name}")
                # Align the test forecast to validation data
                common_idx = np.intersect1d(test_forecast.time_index, val_data.time_index)
                if len(common_idx) > 0:
                    test_forecast = test_forecast.slice(common_idx[0], common_idx[-1])
                    val_data_slice = val_data.slice(common_idx[0], common_idx[-1])
                    if len(test_forecast) > 0 and len(val_data_slice) > 0:
                        test_rmse = rmse(val_data_slice, test_forecast)
                    else:
                        print(f"Error: No overlapping time points for {model_name}")
                        self.test_results[model_name] = float('inf')
                        return
                else:
                    print(f"Error: No common time index for {model_name}")
                    self.test_results[model_name] = float('inf')
                    return
            else:
                # Calculate RMSE
                test_rmse = rmse(val_data, test_forecast)
            
            # Check for NaN in RMSE
            if np.isnan(test_rmse):
                print(f"Warning: NaN RMSE for {model_name}, using infinity")
                test_rmse = float('inf')
            
            # Store test results
            self.test_results[model_name] = test_rmse
            print(f"{model_name} validation RMSE: {test_rmse:.4f}")
            
        except Exception as e:
            print(f"Error calculating RMSE for {model_name}: {str(e)}")
            self.test_results[model_name] = float('inf')  # Use infinity as a fallback

    def evaluate_with_cross_validation(self,
                                     time_series: pd.Series,
                                     n_splits: int = 5,
                                     gap: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Evaluate models using blocked cross-validation.
        
        Args:
            time_series (pd.Series): Input time series data
            n_splits (int): Number of splits for cross-validation
            gap (int): Gap size between train and validation sets
            
        Returns:
            Dictionary containing average RMSE metrics for each model
        """
        folds = self.blocked_cross_validation(time_series, n_splits=n_splits, gap=gap)
        
        # Initialize metrics storage
        model_metrics = {model_name: [] 
                        for model_name in self.models}
        
        for fold_idx, (train_data, val_data, test_data) in enumerate(folds):
            print(f"\nProcessing fold {fold_idx + 1}/{len(folds)}")
            
            for model_name in self.models:
                print(f"Training {model_name}...")
                self.train_with_validation(model_name, train_data, val_data)
                
                # Store metrics for this fold
                metrics = self.test_results[model_name]
                if not np.isnan(metrics) and not np.isinf(metrics):
                    model_metrics[model_name].append(metrics)
        
        # Calculate average metrics
        avg_metrics = {}
        for model_name, metrics_list in model_metrics.items():
            if len(metrics_list) > 0:
                avg_metrics[model_name] = {
                    'avg_rmse': np.mean(metrics_list),
                    'std_rmse': np.std(metrics_list) if len(metrics_list) > 1 else 0
                }
            else:
                avg_metrics[model_name] = {
                    'avg_rmse': float('inf'),
                    'std_rmse': 0
                }
            
        return avg_metrics

    def plot_rmse_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Plot RMSE metrics for all models.
        
        Args:
            metrics (Dict): Dictionary containing RMSE values
        """
        # Filter out infinite or NaN values
        valid_metrics = {k: v for k, v in metrics.items() if not np.isnan(v) and not np.isinf(v)}
        
        if not valid_metrics:
            print("No valid RMSE metrics to plot (all are NaN or Inf)")
            return
            
        models = list(valid_metrics.keys())
        rmse_values = [valid_metrics[m] for m in models]
        
        plt.figure(figsize=(12, 6))
        plt.bar(models, rmse_values)
        plt.title('Model RMSE Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('RMSE')
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'results/model_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.show()

    def train_model(self, model_name, train_data, forecast_horizon=None):
        """
        Train a specific model on the given data.
        
        Args:
            model_name (str): Name of the model to train
            train_data (TimeSeries): Training data
            forecast_horizon (int, optional): Forecast horizon for models that need it
        
        Returns:
            The trained model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        model.fit(train_data)
        return model
    
    def train_all(self, train_data, forecast_horizon=None):
        """
        Train all models on the given data.
        
        Args:
            train_data (TimeSeries): Training data
            forecast_horizon (int, optional): Forecast horizon for models that need it
        """
        for model_name in self.models:
            print(f"Training {model_name} model...")
            self.train_model(model_name, train_data, forecast_horizon)
    
    def forecast(self, model_name, horizon):
        """
        Generate forecast using a specific model.
        
        Args:
            model_name (str): Name of the model to use for forecasting
            horizon (int): Number of steps to forecast
        
        Returns:
            TimeSeries: Forecasted values
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        forecast = model.predict(horizon)
        self.forecasts[model_name] = forecast
        return forecast
    
    def forecast_all(self, horizon):
        """
        Generate forecasts for all models.
        
        Args:
            horizon (int): Number of steps to forecast
        
        Returns:
            dict: Dictionary of model name to forecast
        """
        for model_name in self.models:
            self.forecast(model_name, horizon)
        return self.forecasts
    
    def evaluate(self, model_name, actual_data):
        """
        Evaluate a specific model's forecast against actual data.
        
        Args:
            model_name (str): Name of the model to evaluate
            actual_data (TimeSeries): Actual data to compare against
        
        Returns:
            float: RMSE value
        """
        if model_name not in self.forecasts:
            raise ValueError(f"No forecast found for {model_name}")
        
        forecast = self.forecasts[model_name]
        error = rmse(actual_data, forecast)
        self.results[model_name] = error
        return error
    
    def evaluate_all(self, train_data, test_data, forecast_horizon):
        """
        Evaluate all models using proper training, validation, and test sets.
        
        Args:
            train_data (TimeSeries): Training data
            test_data (TimeSeries): Test data
            forecast_horizon (int): Number of steps to forecast
            
        Returns:
            tuple: (dict of model_name to RMSE, dict of model_name to trained model)
        """
        results = {}
        self.trained_models = {}  # Reset trained models
        benchmark_results = {}  # Store benchmark model results
        ml_results = {}  # Store machine learning model results
        
        # Create model checkpoint directories if they don't exist
        os.makedirs("models", exist_ok=True)
        os.makedirs("models/rnn_checkpoints", exist_ok=True)
        os.makedirs("models/transformer_checkpoints", exist_ok=True)
        os.makedirs("models/tcn_checkpoints", exist_ok=True)
        os.makedirs("models/nbeats_checkpoints", exist_ok=True)
        
        # Validate input data
        if train_data is None or test_data is None:
            raise ValueError("Train or test data is None")
            
        # Check for zero or negative values in the data
        train_values = train_data.values()
        test_values = test_data.values()
        
        if np.any(train_values <= 0) or np.any(test_values <= 0):
            print("\nWARNING: Zero or negative values detected in the stock price data!")
            print("This may cause issues with certain models (especially those using log transforms).")
            print("Consider using adjusted prices or checking data quality.")
            
            # Count and report problematic values
            train_zeros = np.sum(train_values == 0)
            train_neg = np.sum(train_values < 0)
            test_zeros = np.sum(test_values == 0)
            test_neg = np.sum(test_values < 0)
            
            if train_zeros > 0:
                print(f"Training data contains {train_zeros} zero values")
            if train_neg > 0:
                print(f"Training data contains {train_neg} negative values")
            if test_zeros > 0:
                print(f"Test data contains {test_zeros} zero values")
            if test_neg > 0:
                print(f"Test data contains {test_neg} negative values")
                
            # Add small epsilon to zero values to prevent log transform issues
            epsilon = 1e-8
            if np.any(train_values == 0):
                train_values[train_values == 0] = epsilon
                train_data = TimeSeries.from_times_and_values(
                    train_data.time_index,
                    train_values
                )
            if np.any(test_values == 0):
                test_values[test_values == 0] = epsilon
                test_data = TimeSeries.from_times_and_values(
                    test_data.time_index,
                    test_values
                )
            
        # Combine train and test data for proper splitting
        full_data = train_data.append(test_data)
        
        # Split data into train, validation, and test sets
        # Test set: last forecast_horizon points
        # Validation set: previous forecast_horizon points before test
        # Training set: all remaining data
        total_length = len(full_data)
        test_start = total_length - forecast_horizon
        val_start = test_start - forecast_horizon
        
        if val_start <= 0:
            # If not enough data, use a smaller validation set
            val_start = max(test_start // 2, 1)
            
        test_set = full_data[test_start:]
        validation_set = full_data[val_start:test_start]
        training_set = full_data[:val_start]
        
        print(f"\nData splitting:")
        print(f"Training set: {len(training_set)} points ({training_set.start_time()} to {training_set.end_time()})")
        print(f"Validation set: {len(validation_set)} points ({validation_set.start_time()} to {validation_set.end_time()})")
        print(f"Test set: {len(test_set)} points ({test_set.start_time()} to {test_set.end_time()})")
        
        # Check for NaN values in data sets
        for data_set, name in [(training_set, "training"), (validation_set, "validation"), (test_set, "test")]:
            if np.any(np.isnan(data_set.values())):
                print(f"Warning: NaN values detected in {name} data. Filling with forward fill...")
                data_set = data_set.fillna(method='ffill').fillna(method='bfill')
        
        # Apply scaling
        scaler = Scaler()
        scaled_training_set = scaler.fit_transform(training_set)
        scaled_validation_set = scaler.transform(validation_set)
        scaled_test_set = scaler.transform(test_set)
        
        # Process benchmark models first
        benchmark_models = {
            name: model for name, model in self.models.items() 
            if isinstance(model, (NaiveSeasonal, NaiveDrift))
        }
        
        # Process statistical models
        statistical_models = {
            name: model for name, model in self.models.items()
            if isinstance(model, (ExponentialSmoothing, AutoARIMA, Prophet, FFT, AutoCES, AutoETS, AutoMFLES, AutoTBATS, AutoTheta))
            and name not in benchmark_models
        }
        
        # Process machine learning models
        ml_models = {
            name: model for name, model in self.models.items()
            if isinstance(model, (XGBModel, RegressionModel, RNNModel, TCNModel, TransformerModel, NBEATSModel, BlockRNNModel))
        }
        
        print("\n----- Training Benchmark Models -----")
        for name, model in benchmark_models.items():
            try:
                print(f"\nTraining {name}...")
                model.fit(scaled_training_set)
                
                # Generate predictions
                predictions = model.predict(n=len(test_set))
                predictions = scaler.inverse_transform(predictions)
                predictions = TimeSeries.from_times_and_values(
                    test_set.time_index,
                    predictions.values()
                )
                
                # Calculate RMSE
                error = rmse(test_set, predictions)
                if not np.isnan(error) and not np.isinf(error):
                    results[name] = error
                    benchmark_results[name] = error
                    print(f"{name} RMSE: {error:.4f}")
                else:
                    print(f"Warning: {name} produced NaN/Inf RMSE")
                    results[name] = float('inf')
                    
                self.trained_models[name] = model
                
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
                results[name] = float('inf')
        
        print("\n----- Training Statistical Models -----")
        for name, model in statistical_models.items():
            try:
                print(f"\nTraining {name}...")
                model.fit(scaled_training_set)
                
                # Generate predictions
                predictions = model.predict(n=len(test_set))
                predictions = scaler.inverse_transform(predictions)
                predictions = TimeSeries.from_times_and_values(
                    test_set.time_index,
                    predictions.values()
                )
                
                # Calculate RMSE
                error = rmse(test_set, predictions)
                if not np.isnan(error) and not np.isinf(error):
                    results[name] = error
                    print(f"{name} RMSE: {error:.4f}")
                else:
                    print(f"Warning: {name} produced NaN/Inf RMSE")
                    results[name] = float('inf')
                    
                self.trained_models[name] = model
                
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
                results[name] = float('inf')
        
        print("\n----- Training Machine Learning Models -----")
        for name, model in ml_models.items():
            try:
                print(f"\nTraining {name}...")
                
                if isinstance(model, (RNNModel, TCNModel, TransformerModel, NBEATSModel, BlockRNNModel)):
                    # Deep learning models with early stopping on validation set
                    print(f"Training {name} with validation set on {self.device}...")
                    
                    # Train with validation set and early stopping
                    model.fit(
                        scaled_training_set,
                        val_series=scaled_validation_set,
                        verbose=True
                    )
                else:
                    # For other models
                    model.fit(scaled_training_set)
                
                # Generate predictions
                predictions = model.predict(n=len(test_set))
                predictions = scaler.inverse_transform(predictions)
                predictions = TimeSeries.from_times_and_values(
                    test_set.time_index,
                    predictions.values()
                )
                
                # Check for NaN values in predictions
                if np.any(np.isnan(predictions.values())):
                    print(f"Warning: {name} produced NaN values in predictions")
                    predictions = predictions.fillna(method='ffill').fillna(method='bfill')
                    if np.any(np.isnan(predictions.values())):
                        print(f"Error: {name} still contains NaN values after filling")
                        results[name] = float('inf')
                        continue
                
                # Calculate RMSE
                error = rmse(test_set, predictions)
                if not np.isnan(error) and not np.isinf(error):
                    results[name] = error
                    ml_results[name] = error
                    print(f"{name} RMSE: {error:.4f}")
                else:
                    print(f"Warning: {name} produced NaN/Inf RMSE")
                    results[name] = float('inf')
                    
                self.trained_models[name] = model
                
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
                results[name] = float('inf')
        
        # Find best benchmark model RMSE
        if benchmark_results:
            best_benchmark_rmse = min(benchmark_results.values())
            print(f"\nBest benchmark model RMSE: {best_benchmark_rmse:.4f}")
            
            # Find models that outperform the best benchmark
            outperforming_models = {
                name: error for name, error in results.items()
                if error < best_benchmark_rmse and error != float('inf')
            }
            
            if outperforming_models:
                print("\nModels outperforming the best benchmark:")
                for name, error in outperforming_models.items():
                    print(f"- {name}: {error:.4f}")
                
                # Create ensemble model from outperforming models
                self.create_ensemble_model(
                    outperforming_models.keys(), 
                    training_set, 
                    validation_set, 
                    test_set
                )
            else:
                print("\nNo models outperformed the best benchmark.")
        
        return results, self.trained_models
    
    def create_ensemble_model(self, model_names, train_data, val_data, test_data):
        """
        Create an ensemble model from the specified models.
        
        Args:
            model_names (list): Names of models to include in the ensemble
            train_data (TimeSeries): Training data
            val_data (TimeSeries): Validation data
            test_data (TimeSeries): Test data
        """
        print("\nCreating ensemble model...")
        
        # Generate predictions on validation data for each model
        val_predictions = {}
        for name in model_names:
            if name in self.trained_models:
                model = self.trained_models[name]
                try:
                    # Generate predictions
                    predictions = model.predict(n=len(val_data))
                    
                    # If the model uses scaling, inverse transform
                    if isinstance(model, (RNNModel, TCNModel, TransformerModel, NBEATSModel, BlockRNNModel)):
                        scaler = Scaler()
                        scaler.fit(train_data)
                        predictions = scaler.inverse_transform(predictions)
                    
                    # Align time index
                    predictions = TimeSeries.from_times_and_values(
                        val_data.time_index,
                        predictions.values()
                    )
                    
                    # Check for NaN values
                    if not np.any(np.isnan(predictions.values())):
                        val_predictions[name] = predictions
                    else:
                        print(f"Warning: {name} produced NaN values, excluding from ensemble")
                except Exception as e:
                    print(f"Error generating validation predictions for {name}: {str(e)}")
        
        if len(val_predictions) < 2:
            print("Not enough valid models for ensemble, need at least 2")
            return
        
        # Calculate model weights based on inverse RMSE on validation data
        model_weights = {}
        total_inverse_rmse = 0
        for name, predictions in val_predictions.items():
            error = rmse(val_data, predictions)
            if not np.isnan(error) and not np.isinf(error) and error > 0:
                inverse_rmse = 1.0 / error
                model_weights[name] = inverse_rmse
                total_inverse_rmse += inverse_rmse
            else:
                print(f"Warning: Invalid RMSE for {name}, excluding from ensemble")
        
        # Normalize weights
        if total_inverse_rmse > 0:
            for name in model_weights:
                model_weights[name] /= total_inverse_rmse
        
        # Generate ensemble predictions on test data
        test_predictions = {}
        for name in model_weights:
            model = self.trained_models[name]
            try:
                # Generate predictions
                predictions = model.predict(n=len(test_data))
                
                # If the model uses scaling, inverse transform
                if isinstance(model, (RNNModel, TCNModel, TransformerModel, NBEATSModel, BlockRNNModel)):
                    scaler = Scaler()
                    scaler.fit(train_data)
                    predictions = scaler.inverse_transform(predictions)
                
                # Align time index
                predictions = TimeSeries.from_times_and_values(
                    test_data.time_index,
                    predictions.values()
                )
                
                # Check for NaN values
                if not np.any(np.isnan(predictions.values())):
                    test_predictions[name] = predictions
            except Exception as e:
                print(f"Error generating test predictions for {name}: {str(e)}")
        
        # Combine predictions using weighted average
        if test_predictions:
            # Get the shape of predictions
            first_pred = list(test_predictions.values())[0]
            pred_shape = first_pred.values().shape
            
            # Initialize with zeros matching prediction shape
            ensemble_values = np.zeros(pred_shape)
            total_weight = 0
            
            # Weighted sum of predictions
            for name, predictions in test_predictions.items():
                weight = model_weights.get(name, 0)
                # Ensure predictions have the same shape
                if predictions.values().shape == pred_shape:
                    ensemble_values += weight * predictions.values()
                    total_weight += weight
            
            # Normalize if we have valid weights
            if total_weight > 0:
                ensemble_values /= total_weight
            
            # Create ensemble time series
            ensemble_predictions = TimeSeries.from_times_and_values(
                test_data.time_index,
                ensemble_values
            )
            
            # Store in trained models
            class EnsembleModel:
                def __init__(self, models, weights):
                    self.models = models
                    self.weights = weights
                
                def predict(self, n, series=None):
                    predictions = {}
                    for name, model in self.models.items():
                        try:
                            pred = model.predict(n, series)
                            predictions[name] = pred
                        except Exception:
                            pass
                    
                    if not predictions:
                        return None
                    
                    # Get the shape from first prediction
                    first_pred = list(predictions.values())[0]
                    pred_shape = first_pred.values().shape
                    
                    # Initialize with zeros matching prediction shape
                    ensemble_values = np.zeros(pred_shape)
                    total_weight = 0
                    
                    # Weighted sum of predictions
                    for name, pred in predictions.items():
                        weight = self.weights.get(name, 0)
                        # Ensure predictions have the same shape
                        if pred.values().shape == pred_shape:
                            ensemble_values += weight * pred.values()
                            total_weight += weight
                    
                    # Normalize if we have valid weights
                    if total_weight > 0:
                        ensemble_values /= total_weight
                    
                    # Use the time index from the first prediction
                    time_index = first_pred.time_index
                    return TimeSeries.from_times_and_values(time_index, ensemble_values)
            
            # Create and store ensemble model
            ensemble_model = EnsembleModel(
                {name: self.trained_models[name] for name in test_predictions},
                model_weights
            )
            self.trained_models['ensemble'] = ensemble_model
            
            # Calculate ensemble RMSE
            ensemble_error = rmse(test_data, ensemble_predictions)
            if not np.isnan(ensemble_error) and not np.isinf(ensemble_error):
                print(f"\nEnsemble model RMSE: {ensemble_error:.4f}")
                self.results['ensemble'] = ensemble_error
            else:
                print("Warning: Ensemble model produced NaN/Inf RMSE")
    
    def cross_validate_model(self, model_name, time_series, k=5, horizon=30):
        """
        Perform time series cross-validation for a specific model.
        
        Args:
            model_name (str): Name of the model to validate
            time_series (TimeSeries): Complete time series data
            k (int): Number of folds
            horizon (int): Forecast horizon
        
        Returns:
            dict: Dictionary containing RMSE statistics
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        tscv = TimeSeriesSplit(n_splits=k)
        errors = []
        
        # Get time series as pandas DataFrame for splitting
        df = time_series.pd_dataframe()
        
        for train_idx, test_idx in tscv.split(df):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx[:horizon]]  # Take only the horizon length
            
            if len(test_df) < horizon:
                continue  # Skip if test set is too small
            
            try:
                # Convert back to TimeSeries
                train_ts = time_series.pd_series()
                test_ts = time_series.pd_series()
                
                # Check for NaN values
                if np.any(np.isnan(train_ts.values())) or np.any(np.isnan(test_ts.values())):
                    print("Warning: NaN values detected in data, filling...")
                    train_ts = train_ts.fillna(method='ffill').fillna(method='bfill')
                    test_ts = test_ts.fillna(method='ffill').fillna(method='bfill')
                
                # Train and generate predictions
                model.fit(train_ts)
                predictions = model.predict(len(test_ts))
                
                # Check predictions for NaN values
                if np.any(np.isnan(predictions.values())):
                    print("Warning: NaN values in predictions, filling...")
                    predictions = predictions.fillna(method='ffill').fillna(method='bfill')
                
                # Calculate RMSE and check for NaN
                error = rmse(test_ts, predictions)
                
                if not np.isnan(error) and not np.isinf(error):
                    errors.append(error)
                else:
                    print(f"Warning: NaN/Inf RMSE in fold, skipping")
            
            except Exception as e:
                print(f"Error in cross-validation fold: {str(e)}")
                continue
        
        # Calculate statistics
        if len(errors) > 0:
            result = {
                'rmse': np.mean(errors),
                'rmse_std': np.std(errors) if len(errors) > 1 else 0
            }
        else:
            result = {
                'rmse': float('inf'),
                'rmse_std': 0
            }
        
        return result
    
    def plot_forecasts(self, train_data, test_data, horizon):
        """
        Plot the forecasts from all models against the actual data.
        
        Args:
            train_data (TimeSeries): Training data
            test_data (TimeSeries): Test data
            horizon (int): Forecast horizon
        """
        plt.figure(figsize=(15, 8))
        
        # Plot train and test data
        train_data.plot(label='Train')
        test_data.plot(label='Test')
        
        # Plot forecasts
        for model_name, forecast in self.forecasts.items():
            if model_name in self.results:
                rmse_value = self.results[model_name]
                forecast.plot(label=f'{model_name} (RMSE: {rmse_value:.4f})')
            else:
                forecast.plot(label=model_name)
        
        plt.title(f'Stock Price Forecasts (Horizon: {horizon} days)')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig(f'results/forecasts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.show() 