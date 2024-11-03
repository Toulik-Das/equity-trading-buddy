import streamlit as st
import openai
import requests
import pandas as pd
import numpy as np
import os

from utils.stockSignal import *

BASE_URL = 'https://www.alphavantage.co/query'

def get_stock_symbol(company_name, alpha_vantage_api_key):
    """Fetch the stock symbol for a given company name."""
    url = f"{BASE_URL}?function=SYMBOL_SEARCH&keywords={company_name}&apikey={alpha_vantage_api_key}"
    response = requests.get(url)
    data = response.json()
    if 'bestMatches' in data and len(data['bestMatches']) > 0:
        return data['bestMatches'][0]['1. symbol']
    else:
        return None

def get_company_info(company_name):
    """Fetch company logo and description from Wikipedia."""
    search_url = f"https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": company_name,
        "prop": "extracts",
        "exintro": True,  # Get only the introductory section
        "explaintext": True  # Return plain text instead of HTML
    }
    response = requests.get(search_url, params=params)
    data = response.json()

    pages = data.get("query", {}).get("pages", {})
    if pages:
        # Get the first page in the result
        page = next(iter(pages.values()))
        description = page.get("extract", "No description available.")
        
        # Use Clearbit logo API to fetch company logo
        logo_url = f"https://logo.clearbit.com/{company_name.lower().replace(' ', '')}.com"
        
        return logo_url, description
    else:
        return None, "Company not found."

def main():
    # Set the page configuration
    st.set_page_config(
        page_title="Stock Analysis Tool",
        page_icon="📈",
        layout="wide",
    )

    # Title of the app
    st.title("Stock Analysis Tool")

    # Sidebar for user inputs
    st.sidebar.header("User Input")

    # Inputs for Company Name and API keys
    company_name = st.sidebar.text_input("Company Name", placeholder="Enter company name here...")
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", placeholder="Enter your OpenAI API Key...")
    alpha_vantage_api_key = st.sidebar.text_input("Alpha Vantage API Key", type="password", placeholder="Enter your Alpha Vantage API Key...")

    # Create a layout for buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Get OpenAI API Key"):
            st.markdown("[Create OpenAI API Key](https://platform.openai.com/signup)", unsafe_allow_html=True)
    with col2:
        if st.button("Get Alpha Vantage API Key"):
            st.markdown("[Create Alpha Vantage API Key](https://www.alphavantage.co/support/#api-key)", unsafe_allow_html=True)

    # Display company logo and description
    if company_name:
        logo_url, description = get_company_info(company_name)
        if logo_url:
            st.markdown(
                f"""
                <div style="text-align: left; padding: 10px;">
                    <img src="{logo_url}" alt="{company_name} Logo" style="width:100px; border: 2px solid #ddd; border-radius: 8px; padding: 5px;">
                </div>
                """,
                unsafe_allow_html=True
            )
            #st.write(f"### About {company_name}:")
            #st.markdown(f"<p style='color: white;'>{description}</p>", unsafe_allow_html=True)  # Change text color to white
        else:
            st.error(description)

    # Submit button to trigger analysis
    if st.sidebar.button("Get Recommendation"):
        # Set OpenAI API key
        openai.api_key = openai_api_key

        # Fetch stock symbol
        stock_symbol = get_stock_symbol(company_name, alpha_vantage_api_key)
        if stock_symbol is None:
            st.error("Stock symbol not found. Please check the company name.")
        else:
            st.success(f"Stock symbol for **{company_name}**: `{stock_symbol}`")

            # Fetch stock data
            df = get_stock_data(stock_symbol, alpha_vantage_api_key)
            if df is not None:
                # Calculate indicators
                df = calculate_macd(df)
                df = calculate_rsi(df)
                df = calculate_bollinger_bands(df)

                # Generate trading signals
                df = generate_signals(df)

                # Get recommendation with GPT-4
                recommendation = get_recommendation(df, stock_symbol)
                st.write("### Final Recommendation:")
                placeholder = st.empty()  # Create an empty placeholder
                for part in recommendation:
                    print(part)
                    placeholder.markdown(part)  # Stream each part of the recommendation
            else:
                st.error("Failed to retrieve stock data.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
