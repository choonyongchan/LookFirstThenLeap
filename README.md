# Stock Price Forecasting with Machine Learning and Sentiment Analysis

A comprehensive system for analyzing and forecasting stock prices using time series analysis, machine learning models, and sentiment analysis.

## Features

- Download and analyze stock data from Yahoo Finance
- Benchmark forecasting methods (naive, average, drift)
- Statistical modeling (ARIMA, Theta, etc.)
- Machine learning and deep learning models (RNNs, TCNs, Transformers, N-BEATS)
- Financial news sentiment analysis for stock tickers from NewsAPI
- Fear & Greed index integration
- Combined models that leverage both price data and sentiment
- Detailed evaluation with multiple metrics (RMSE, MAPE, MAE)
- Cross-validation for hyperparameter tuning
- Visualizations and HTML reports

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/stock-forecasting.git
cd stock-forecasting
```

2. Create a virtual environment and install dependencies:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. (Optional) Set up NewsAPI credentials. For sentiment analysis to work with real news data, set this environment variable:
```
export NEWS_API_KEY=your_newsapi_key
```

If not provided, the system will generate synthetic sentiment data for demonstration.

## Usage

### Basic Forecasting

Run the main script to analyze a stock using all available models:

```
python stock_forecaster.py
```

By default, this will analyze UOB stock (U11.SI) from Yahoo Finance. 

### Customizing Analysis

You can modify the `stock_forecaster.py` file to analyze different stocks or timeframes by changing these parameters:

```python
ticker = "U11.SI"  # Change this to any Yahoo Finance ticker
start_date = "2018-01-01"  # Change start date
forecast_horizon = 30  # Change forecast length
```

### Output

The script will:
1. Download or load stock data
2. Train multiple forecasting models
3. Generate forecasts
4. Evaluate model performance using RMSE
5. Create visualizations
6. Analyze sentiment (real or synthetic)
7. Generate a combined forecast
8. Save all results to the `results` directory

## Project Structure

- `stock_forecaster.py`: Main entry script
- `src/`: Source code modules
  - `data_loader.py`: Downloads and processes stock data
  - `forecasting.py`: Implements various forecasting models
  - `sentiment_analysis.py`: Analyzes Twitter sentiment and Fear & Greed index
  - `combined_model.py`: Combines price forecasting with sentiment analysis
  - `utils.py`: Utility functions for metrics, plotting, etc.
- `data/`: Cached data files
- `results/`: Output files, plots, and reports
- `models/`: Saved models

## Advanced Usage

### Adding New Models

You can add new models by extending the `Forecaster` class in `src/forecasting.py`. For example:

```python
def add_new_model(self):
    self.models['new_model'] = NewForecastingModel(params...)
```

### Custom Sentiment Analysis

To implement a custom sentiment analysis approach, modify the `SentimentAnalyzer` class in `src/sentiment_analysis.py`.

### Cross-Validation

For models that require hyperparameter tuning, use the built-in cross-validation:

```python
forecaster = Forecaster()
best_rmse = forecaster.cross_validate_model('model_name', stock_data, k=5)
```

## License

MIT

## Disclaimer

This software is for educational purposes only. It is not financial advice, and should not be used for making investment decisions. Past performance is not indicative of future results. 