#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for sentiment analysis of stock-related data.
"""

import pandas as pd
import numpy as np
import os
import requests
import json
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import re
from darts import TimeSeries

class SentimentAnalyzer:
    """
    Analyzes sentiment data from various sources for stock market prediction.
    """
    
    def __init__(self):
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Set up NewsAPI credentials - Replace with your own API key or use environment variable
        self.news_api_key = "55c2568c10a74ec89c8a237b36e8546d"#"0c8a89ec1e89436a9586b07c1d29b4f4"#os.environ.get('NEWS_API_KEY') 
        self.news_api_auth = self.news_api_key is not None
        if not self.news_api_auth:
            print("NewsAPI key not found in environment variables (NEWS_API_KEY)")
    
    def analyze_sentiment(self, ticker, days=30, force_download=False, start_date=None, end_date=None):
        """
        Analyze news sentiment for a specific ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            days (int): Number of days to analyze
            force_download (bool): Force download even if file exists
            start_date (datetime, optional): Start date for analysis
            end_date (datetime, optional): End date for analysis
            
        Returns:
            TimeSeries: Darts TimeSeries object with sentiment scores
        """
        file_path = os.path.join(self.data_dir, f"{ticker.replace('.', '_')}_sentiment.csv")
        
        if os.path.exists(file_path) and not force_download:
            print(f"Loading {ticker} sentiment data from file")
            sentiment_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        else:
            print(f"Generating {ticker} sentiment data")
            
            # If NewsAPI auth is available, use NewsAPI
            if self.news_api_auth:
                sentiment_df = self._get_news_sentiment(ticker, days, start_date, end_date)
            else:
                # Fallback to generating synthetic sentiment data
                sentiment_df = self._generate_synthetic_sentiment(ticker, days, start_date, end_date)
            
            sentiment_df.to_csv(file_path)
        
        # Ensure index is timezone-naive and contains only dates
        sentiment_df.index = pd.to_datetime(sentiment_df.index).normalize()
        
        # Create Darts TimeSeries from the sentiment scores
        ts = TimeSeries.from_dataframe(sentiment_df)
        return ts
    
    def _get_news_sentiment(self, ticker, days, start_date=None, end_date=None):
        """
        Get news sentiment for a specific ticker using NewsAPI.
        
        Args:
            ticker (str): Stock ticker symbol
            days (int): Number of days to analyze
            start_date (datetime, optional): Start date for analysis
            end_date (datetime, optional): End date for analysis
            
        Returns:
            DataFrame: DataFrame with sentiment scores
        """
        if end_date is None:
            end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if start_date is None:
            start_date = end_date - timedelta(days=days)
            
        # Ensure dates are timezone-naive and normalized
        end_date = pd.to_datetime(end_date).normalize()
        start_date = pd.to_datetime(start_date).normalize()
        
        # Create date range for daily sentiment analysis
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        sentiment_df = pd.DataFrame(index=date_range, columns=['compound', 'positive', 'negative', 'neutral', 'volume'])
        sentiment_df.fillna({'compound': 0, 'positive': 0, 'negative': 0, 'neutral': 0, 'volume': 0}, inplace=True)
        
        # Remove any extension from ticker for searching
        search_ticker = ticker.split('.')[0]
        
        # Company name mapping for better search results (expand as needed)
        company_name_mapping = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google',
            'AMZN': 'Amazon',
            'META': 'Meta OR Facebook',
            'TSLA': 'Tesla',
            'NVDA': 'Nvidia',
            'U11': 'UOB OR "United Overseas Bank"',
            'D05': 'DBS OR "DBS Bank" OR "Development Bank of Singapore"',
        }
        
        # Get company name for search, if available, otherwise use ticker
        search_term = company_name_mapping.get(search_ticker, search_ticker)
        
        # Calculate chunk of days to fetch (NewsAPI has limitations)
        chunk_size = 7  # NewsAPI free tier has 7 day chunks
        chunks = [(start_date + timedelta(days=i), min(start_date + timedelta(days=i + chunk_size - 1), end_date)) 
                 for i in range(0, (end_date - start_date).days + 1, chunk_size)]
        
        all_articles = []
        
        for chunk_start, chunk_end in chunks:
            try:
                # Format dates for API
                from_date = chunk_start.strftime('%Y-%m-%d')
                to_date = chunk_end.strftime('%Y-%m-%d')
                
                # Construct URL for NewsAPI
                url = (f'https://newsapi.org/v2/everything?'
                      f'q={search_term}&'
                      f'from={from_date}&'
                      f'to={to_date}&'
                      f'language=en&'
                      f'sortBy=publishedAt&'
                      f'apiKey={self.news_api_key}')
                
                # Make the request
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    if data['status'] == 'ok':
                        articles = data.get('articles', [])
                        all_articles.extend(articles)
                        print(f"Retrieved {len(articles)} articles for {search_term} from {from_date} to {to_date}")
                    else:
                        print(f"API Error: {data.get('message', 'Unknown error')}")
                else:
                    print(f"Request Error: Status code {response.status_code}")
                    
                # Respect API rate limits
                import time
                time.sleep(0.5)
                    
            except Exception as e:
                print(f"Error fetching news for {search_term} from {from_date} to {to_date}: {e}")
        
        # Process articles by date
        date_articles = {}
        for article in all_articles:
            try:
                # Convert timestamp to date only (remove time component)
                pub_date = datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ').date()
                if pub_date not in date_articles:
                    date_articles[pub_date] = []
                date_articles[pub_date].append(article)
            except (ValueError, KeyError) as e:
                print(f"Error processing article date: {e}")
        
        # Analyze sentiment for each day
        for date in date_range:
            date_key = date.date()
            articles = date_articles.get(date_key, [])
            
            # Initialize sentiment counters
            compound_sum = 0
            positive_sum = 0
            negative_sum = 0
            neutral_sum = 0
            count = 0
            
            # Analyze sentiment for each article
            for article in articles:
                # Combine title and description for analysis
                title = article.get('title', '') or ''
                description = article.get('description', '') or ''
                text = f"{title} {description}".strip()
                
                # Clean the text
                text = self._clean_text(text)
                
                # Skip if empty
                if not text:
                    continue
                    
                # Get sentiment scores
                sentiment = self.sentiment_analyzer.polarity_scores(text)
                
                compound_sum += sentiment['compound']
                positive_sum += sentiment['pos']
                negative_sum += sentiment['neg']
                neutral_sum += sentiment['neu']
                count += 1
            
            # Calculate average sentiment scores
            if count > 0:
                sentiment_df.loc[date, 'compound'] = compound_sum / count
                sentiment_df.loc[date, 'positive'] = positive_sum / count
                sentiment_df.loc[date, 'negative'] = negative_sum / count
                sentiment_df.loc[date, 'neutral'] = neutral_sum / count
                sentiment_df.loc[date, 'volume'] = count
        
        # Ensure no NaN values in the DataFrame
        sentiment_df.fillna({'compound': 0, 'positive': 0, 'negative': 0, 'neutral': 0, 'volume': 0}, inplace=True)
        
        # Ensure index is timezone-naive and contains only dates
        sentiment_df.index = pd.to_datetime(sentiment_df.index).normalize()
        
        return sentiment_df
    
    def _generate_synthetic_sentiment(self, ticker, days, start_date=None, end_date=None):
        """
        Generate synthetic sentiment data for demonstration purposes.
        
        Args:
            ticker (str): Stock ticker symbol
            days (int): Number of days to generate
            start_date (datetime, optional): Start date for analysis
            end_date (datetime, optional): End date for analysis
            
        Returns:
            DataFrame: DataFrame with synthetic sentiment scores
        """
        print("Generating synthetic sentiment data (NewsAPI key not available)")
        if end_date is None:
            end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if start_date is None:
            start_date = end_date - timedelta(days=days)
            
        # Ensure dates are timezone-naive and normalized
        end_date = pd.to_datetime(end_date).normalize()
        start_date = pd.to_datetime(start_date).normalize()
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate random sentiment data with a slight trend
        np.random.seed(42)  # For reproducibility
        
        # Generate trend component
        t = np.linspace(0, 1, len(date_range))
        trend = 0.1 * np.sin(2 * np.pi * t)
        
        # Generate random component
        noise = np.random.normal(0, 0.2, len(date_range))
        
        # Combine trend and noise
        compound = trend + noise
        compound = np.clip(compound, -1, 1)  # Ensure values are between -1 and 1
        
        # Calculate positive and negative components based on compound
        positive = (compound + 1) / 2
        negative = 1 - positive
        neutral = np.random.uniform(0.3, 0.7, len(date_range))
        
        # Normalize to ensure all components sum to 1
        total = positive + negative + neutral
        positive = positive / total
        negative = negative / total
        neutral = neutral / total
        
        # Generate random article volume
        volume = np.random.randint(5, 30, len(date_range))
        
        # Create DataFrame
        sentiment_df = pd.DataFrame({
            'compound': compound,
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'volume': volume
        }, index=date_range)
        
        return sentiment_df
    
    def _clean_text(self, text):
        """
        Clean text by removing URLs, special characters, etc.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str) or not text.strip():
            return ""
            
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters but keep punctuation for sentiment
        text = re.sub(r'[^a-zA-Z0-9\s\.,;:!?\'"-]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_fear_greed_index(self, days=30, force_download=False, start_date=None, end_date=None):
        """
        Get historical Fear & Greed Index data.
        
        Args:
            days (int): Number of days of data to retrieve
            force_download (bool): Force download even if file exists
            start_date (datetime, optional): Start date for analysis
            end_date (datetime, optional): End date for analysis
            
        Returns:
            TimeSeries: Darts TimeSeries object with Fear & Greed Index data
        """
        file_path = os.path.join(self.data_dir, "fear_greed_index.csv")
        
        if os.path.exists(file_path) and not force_download:
            print("Loading Fear & Greed Index data from file")
            fg_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        else:
            print("Generating Fear & Greed Index data")
            fg_df = self._get_fear_greed_data(days, start_date, end_date)
            fg_df.to_csv(file_path)
        
        # Ensure dates are timezone-naive and normalized
        fg_df.index = pd.to_datetime(fg_df.index).normalize()
        
        # Trim the data to match the requested date range
        if start_date is not None:
            start_date = pd.to_datetime(start_date).normalize()
            fg_df = fg_df[fg_df.index >= start_date]
        if end_date is not None:
            end_date = pd.to_datetime(end_date).normalize()
            fg_df = fg_df[fg_df.index <= end_date]
        
        # Create Darts TimeSeries from the Fear & Greed Index
        ts = TimeSeries.from_dataframe(fg_df, value_cols=['value'])
        return ts
    
    def _get_fear_greed_data(self, days, start_date=None, end_date=None):
        """
        Get Fear & Greed Index data from API or generate synthetic data.
        
        Args:
            days (int): Number of days of data to retrieve
            start_date (datetime, optional): Start date for analysis
            end_date (datetime, optional): End date for analysis
            
        Returns:
            DataFrame: DataFrame with Fear & Greed Index data
        """
        try:
            # Try to get data from CNN Fear & Greed API (if it exists)
            url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                scores = data.get('fear_and_greed', {}).get('score', [])
                
                dates = []
                values = []
                categories = []
                
                for item in scores:
                    # Convert timestamp to date only
                    date = pd.to_datetime(item.get('timestamp')).normalize()
                    if start_date is not None and date < pd.to_datetime(start_date).normalize():
                        continue
                    if end_date is not None and date > pd.to_datetime(end_date).normalize():
                        continue
                    dates.append(date)
                    values.append(item.get('score'))
                    categories.append(item.get('rating'))
                
                # Create DataFrame
                fg_df = pd.DataFrame({
                    'value': values,
                    'category': categories
                }, index=pd.to_datetime(dates))
                
                # Sort by date
                fg_df.sort_index(inplace=True)
                
                return fg_df
            else:
                # Fall back to synthetic data
                return self._generate_synthetic_fear_greed(days, start_date, end_date)
                
        except Exception as e:
            print(f"Error getting Fear & Greed Index data: {e}")
            return self._generate_synthetic_fear_greed(days, start_date, end_date)
    
    def _generate_synthetic_fear_greed(self, days, start_date=None, end_date=None):
        """
        Generate synthetic Fear & Greed Index data.
        
        Args:
            days (int): Number of days of data to generate
            start_date (datetime, optional): Start date for analysis
            end_date (datetime, optional): End date for analysis
            
        Returns:
            DataFrame: DataFrame with synthetic Fear & Greed Index data
        """
        print("Generating synthetic Fear & Greed Index data")
        
        # Use provided dates or calculate from days parameter
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=days)
            
        # Ensure dates are timezone-naive and normalized
        end_date = pd.to_datetime(end_date).normalize()
        start_date = pd.to_datetime(start_date).normalize()
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate random Fear & Greed values with trend and some mean-reversion
        np.random.seed(42)  # For reproducibility
        
        # Start with a random value between 20 and 80
        value = np.random.randint(20, 80)
        values = [value]
        
        # Generate the rest with some mean-reversion
        for _ in range(len(date_range) - 1):
            # Mean-reversion component (50 is neutral)
            mean_reversion = 0.1 * (50 - value)
            
            # Random component
            noise = np.random.normal(0, 5)
            
            # Update value
            value = value + mean_reversion + noise
            
            # Ensure value is within valid range
            value = max(0, min(100, value))
            
            values.append(value)
        
        # Assign categories based on value ranges
        categories = []
        for v in values:
            if v <= 25:
                categories.append('Extreme Fear')
            elif v <= 35:
                categories.append('Fear')
            elif v <= 50:
                categories.append('Neutral')
            elif v <= 75:
                categories.append('Greed')
            else:
                categories.append('Extreme Greed')
        
        # Create DataFrame
        fg_df = pd.DataFrame({
            'value': values,
            'category': categories
        }, index=date_range)
        
        return fg_df
    
    def plot_sentiment(self, sentiment_data, ticker):
        """
        Plot sentiment data.
        
        Args:
            sentiment_data (TimeSeries): Sentiment time series
            ticker (str): Stock ticker symbol
        """
        plt.figure(figsize=(15, 10))
        
        # Convert to DataFrame for plotting
        df = sentiment_data.pd_dataframe()
        
        # Plot compound sentiment
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df['compound'], label='Compound Sentiment', color='blue')
        plt.title(f'Financial News Sentiment for {ticker}')
        plt.axhline(y=0, color='gray', linestyle='--')
        plt.ylabel('Compound Score')
        plt.grid(True)
        plt.legend()
        
        # Plot volume
        plt.subplot(2, 1, 2)
        plt.bar(df.index, df['volume'], label='Article Volume', color='green', alpha=0.7)
        plt.title(f'News Article Volume for {ticker}')
        plt.ylabel('Volume')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'results/{ticker.replace(".", "_")}_sentiment.png')
        plt.show()
    
    def plot_fear_greed(self, fear_greed_data):
        """
        Plot Fear & Greed Index data.
        
        Args:
            fear_greed_data (TimeSeries): Fear & Greed Index time series
        """
        plt.figure(figsize=(15, 6))
        
        # Convert to DataFrame for plotting
        df = fear_greed_data.pd_dataframe()
        
        # Create colormap based on values
        colors = []
        for value in df['value']:
            if value <= 25:
                colors.append('red')
            elif value <= 35:
                colors.append('orange')
            elif value <= 50:
                colors.append('yellow')
            elif value <= 75:
                colors.append('lightgreen')
            else:
                colors.append('green')
        
        # Plot Fear & Greed Index
        plt.bar(df.index, df['value'], color=colors, alpha=0.7)
        plt.title('Fear & Greed Index')
        plt.ylabel('Index Value')
        plt.axhline(y=50, color='gray', linestyle='--')
        
        # Add horizontal lines for category boundaries
        plt.axhline(y=25, color='gray', linestyle=':')
        plt.axhline(y=35, color='gray', linestyle=':')
        plt.axhline(y=75, color='gray', linestyle=':')
        
        # Add text labels for categories
        plt.text(df.index[0], 12.5, 'Extreme Fear', ha='left', va='center')
        plt.text(df.index[0], 30, 'Fear', ha='left', va='center')
        plt.text(df.index[0], 42.5, 'Neutral', ha='left', va='center')
        plt.text(df.index[0], 62.5, 'Greed', ha='left', va='center')
        plt.text(df.index[0], 87.5, 'Extreme Greed', ha='left', va='center')
        
        plt.grid(True)
        plt.savefig('results/fear_greed_index.png')
        plt.show() 