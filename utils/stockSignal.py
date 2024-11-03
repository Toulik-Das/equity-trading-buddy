import openai
import requests
import pandas as pd
import numpy as np
import os

BASE_URL = 'https://www.alphavantage.co/query'


def get_stock_data(symbol,ALPHA_VANTAGE_API_KEY):
    """Fetch daily stock data for a given symbol."""
    function = 'TIME_SERIES_DAILY'
    url = f"{BASE_URL}?function={function}&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}&outputsize=full"
    response = requests.get(url, verify=False)
    data = response.json()
    timeseries_key = 'Time Series (Daily)'

    if timeseries_key in data:
        df = pd.DataFrame(data[timeseries_key]).T
        df = df.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        }).astype(float)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    else:
        print("Error fetching data:", data)
        return None

def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    """Calculate MACD line and Signal line."""
    df['EMA_short'] = df['close'].ewm(span=short_window, adjust=False).mean()
    df['EMA_long'] = df['close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = df['EMA_short'] - df['EMA_long']
    df['Signal'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    return df

def calculate_rsi(df, period=14):
    """Calculate the Relative Strength Index (RSI)."""
    delta = df['close'].diff(1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_bollinger_bands(df, window=20):
    """Calculate Bollinger Bands."""
    df['SMA'] = df['close'].rolling(window=window).mean()
    df['STD'] = df['close'].rolling(window=window).std()
    df['Upper_Band'] = df['SMA'] + (2 * df['STD'])
    df['Lower_Band'] = df['SMA'] - (2 * df['STD'])
    return df

def generate_signals(df):
    """Generate Buy/Sell signals based on MACD, RSI, and Bollinger Bands."""
    # MACD Signals
    df['MACD_Buy'] = (df['MACD'] > df['Signal']) & (df['MACD'].shift(1) <= df['Signal'].shift(1))
    df['MACD_Sell'] = (df['MACD'] < df['Signal']) & (df['MACD'].shift(1) >= df['Signal'].shift(1))

    # RSI Signals
    df['RSI_Buy'] = df['RSI'] < 30
    df['RSI_Sell'] = df['RSI'] > 70

    # Bollinger Bands Signals
    df['Bollinger_Buy'] = df['close'] < df['Lower_Band']
    df['Bollinger_Sell'] = df['close'] > df['Upper_Band']

    # Composite Signal
    df['Buy_Signal'] = df['MACD_Buy'] & df['RSI_Buy'] & df['Bollinger_Buy']
    df['Sell_Signal'] = df['MACD_Sell'] & df['RSI_Sell'] & df['Bollinger_Sell']
    return df

def analyze_sentiment(symbol):
    """Use GPT-4 to analyze recent stock-related news sentiment."""
    # Simulate fetching recent news or use placeholder text
    recent_news = f"Latest news about {symbol}: strong quarterly earnings, product launch success, and market expansion plans."

    # Construct the prompt
    prompt = f"Analyze the sentiment of the following news and determine if it supports a Buy, Sell, or Hold position for {symbol}: {recent_news}"

    # GPT-4 query using ChatCompletion
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500
    )

    sentiment = response.choices[0].message.content.strip()  # Note the dot notation
    
    return sentiment


def get_recommendation(df, symbol):
    """Check the last row for composite buy/sell signals and adjust with GPT-4 sentiment analysis."""
    last_row = df.iloc[-1]
    indicator_signal = "Buy" if last_row['Buy_Signal'] else "Sell" if last_row['Sell_Signal'] else "Hold"

    # Get GPT-4 sentiment analysis
    sentiment = analyze_sentiment(symbol)

    # Initialize the explanation based on technical indicators
    explanation = []

    # Determine the base explanation
    if indicator_signal == "Buy":
        explanation.append("The technical indicators strongly suggest a buying opportunity.")
    elif indicator_signal == "Sell":
        explanation.append("The technical indicators indicate a potential selling opportunity.")
    elif indicator_signal == "Hold":
        explanation.append("The technical indicators recommend holding the position for now.")

    # Create a prompt for GPT-4 to generate a more detailed explanation
    prompt = (f"Please provide a comprehensive and engaging explanation for a stock recommendation based on the following:\n"
              f"Symbol: {symbol}\n"
              f"Indicator Signal: {indicator_signal}\n"
              f"Sentiment Analysis: {sentiment}\n"
              f"Explain the reasoning behind the recommendation, highlighting key insights, market trends, and the potential implications for an investor. Make it clear and accessible for a general audience.")

    # Get explanation from GPT-4
    gpt_response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500
    )
    
    detailed_explanation = gpt_response.choices[0].message.content.strip()

    # Include future prediction and probability analysis
    prediction_prompt = (f"Based on the current market analysis, please provide a future prediction for the stock {symbol}, "
                         f"including the expected percentage increase or decrease over the next 1 to 3 months, and the probability of reaching buy/sell thresholds. "
                         f"Also, suggest an optimal time frame for buying or selling if the current recommendation is to hold. Make it convincing, clear and accessible for a general audience.")

    # Get prediction and probability from GPT-4
    prediction_response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prediction_prompt}],
        max_tokens=1500
    )

    prediction_details = prediction_response.choices[0].message.content.strip()

    # Recommendations based on risk tolerance
    risk_based_recommendations = {
        "Low Risk": f"For conservative investors, it's recommended to maintain your current position and monitor closely. With the current sentiment, a hold strategy may lead to moderate growth without taking on additional risk.",
        "Medium Risk": f"For moderate risk investors, considering a partial buy or reinvestment during dips might be beneficial, particularly if the stock dips by around 5-10%. This strategy could yield a better return while still balancing risk.",
        "High Risk": f"For aggressive investors, this may be an opportune time to buy more shares, especially if the price decreases by 10% or more. The technical indicators and positive sentiment suggest potential for significant gains."
    }

    # Combine the base explanation with the detailed explanation and prediction
    full_explanation = (
        f"### Recommendation: {indicator_signal}\n\n"  # Clearly states the recommendation based on the indicators
        f"### Summary of Analysis:\n{detailed_explanation}\n\n"  # Provides a detailed breakdown of the analysis conducted by GPT-4
        f"### Key Takeaways:\n{explanation[0]}\n\n"  # Highlights the primary takeaway from the technical indicators
        f"### Future Prediction:\n{prediction_details}\n\n"  # Outlines the expected future performance and suggestions for action
        f"### Recommendations Based on Risk Tolerance:\n"  # Introduces the risk tolerance recommendations
        f"- **Low Risk:** {risk_based_recommendations['Low Risk']}\n"
        f"- **Medium Risk:** {risk_based_recommendations['Medium Risk']}\n"
        f"- **High Risk:** {risk_based_recommendations['High Risk']}\n"
    )

    # Yield each section for streaming
    # yield f"### Recommendation: {indicator_signal}\n\n"  # Recommendation
    # yield f"### Summary of Analysis:\n{detailed_explanation}\n\n"  # Detailed analysis
    # yield f"### Key Takeaways:\n{explanation[0]}\n\n"  # Key takeaway
    # yield f"### Future Prediction:\n{prediction_details}\n\n"  # Future prediction
    # yield f"### Recommendations Based on Risk Tolerance:\n"  # Risk recommendations
    # yield f"- **Low Risk:** {risk_based_recommendations['Low Risk']}\n"
    # yield f"- **Medium Risk:** {risk_based_recommendations['Medium Risk']}\n"
    # yield f"- **High Risk:** {risk_based_recommendations['High Risk']}\n"

    return full_explanation


def get_stock_symbol(company_name,ALPHA_VANTAGE_API_KEY):
    """Fetch the stock symbol for a given company name."""
    url = f"{BASE_URL}?function=SYMBOL_SEARCH&keywords={company_name}&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url)
    data = response.json()
    if 'bestMatches' in data and len(data['bestMatches']) > 0:
        return data['bestMatches'][0]['1. symbol']
    else:
        return None
    
# Main function
def main(stock_symbol):
    # Fetch stock data
    df = get_stock_data(stock_symbol)
    if df is None:
        return

    # Calculate indicators
    df = calculate_macd(df)
    df = calculate_rsi(df)
    df = calculate_bollinger_bands(df)

    # Generate trading signals
    df = generate_signals(df)

    # Get recommendation with GPT-4
    recommendation = get_recommendation(df, stock_symbol)
    print(f"Final Recommendation for {stock_symbol}: {recommendation}")
    return recommendation
