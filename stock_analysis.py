import datetime as dt
from matplotlib import dates, ticker
import numpy as np
# from util import get_data
import matplotlib.pyplot as plt
import pandas as pd
import os  		 
import yfinance as yf


'''
Code implementing indicators as functions that operate on DataFrames.
Name: Apurva Gandhi
'''
# Helper Function
def calculate_rolling_mean(values, window_size):
    return values.rolling(window_size).mean()

# Helper Function
def calculate_rolling_std(values, window_size):
    return values.rolling(window_size).std()

def symbol_to_path(symbol, base_dir=None):  		  	   		  		 		  		  		    	 		 		   		 		  
    """Return CSV file path given ticker symbol."""  		  	   		  		 		  		  		    	 		 		   		 		  
    if base_dir is None:  		  	   		  		 		  		  		    	 		 		   		 		  
        base_dir = os.environ.get("MARKET_DATA_DIR", "Data/")  		  	   		  		 		  		  		    	 		 		   		 		  
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))  	

def get_stock_data_from_yahoo(symbols, start_date="2022-10-25", end_date="2022-10-29"):
    dates = pd.date_range(start_date, end_date)
    stock_data = pd.DataFrame(index=dates)  		  	   		  		 		  		  		    	 		 		   		 		  
    for symbol in symbols:
        # Get the stock data
        stock_data_temp = yf.download(symbol, start=start_date, end=end_date)
        # Select only the "Adj Close" column
        stock_data_temp = stock_data_temp[["Adj Close"]]
        # Rename the "Adj Close" column to the symbol name
        stock_data_temp = stock_data_temp.rename(columns={"Adj Close": symbol})
        stock_data = stock_data.join(stock_data_temp)
    stock_data = stock_data.dropna()
    return stock_data
	  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		    	   		  		 		  		  		    	 		 		   		 		  
def get_data(symbols, dates, addSPY=True, colname="Adj Close"):  		  	   		  		 		  		  		    	 		 		   		 		  
    """Read stock data (adjusted close) for given symbols from CSV files."""  		  	   		  		 		  		  		    	 		 		   		 		  
    df = pd.DataFrame(index=dates)  		  	   		  		 		  		  		    	 		 		   		 		  
    if addSPY and "SPY" not in symbols:  # add SPY for reference, if absent  		  	   		  		 		  		  		    	 		 		   		 		  
        symbols = ["SPY"] + list(  		  	   		  		 		  		  		    	 		 		   		 		  
            symbols  		  	   		  		 		  		  		    	 		 		   		 		  
        )  # handles the case where symbols is np array of 'object'  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    for symbol in symbols:  		  	   		  		 		  		  		    	 		 		   		 		  
        df_temp = pd.read_csv(  		  	   		  		 		  		  		    	 		 		   		 		  
            symbol_to_path(symbol),  		  	   		  		 		  		  		    	 		 		   		 		  
            index_col="Date",  		  	   		  		 		  		  		    	 		 		   		 		  
            parse_dates=True,  		  	   		  		 		  		  		    	 		 		   		 		  
            usecols=["Date", colname],  		  	   		  		 		  		  		    	 		 		   		 		  
            na_values=["nan"],  		  	   		  		 		  		  		    	 		 		   		 		  
        )  
        df_temp = df_temp.rename(columns={colname: symbol})  	
        df = df.join(df_temp) 		  	   		  		 		  		  		    	 		 		   		 		  
    return df_temp  	

# Indicator 1 - Simple Moving Average
def calculate_simple_moving_average(prices, window_size = 10):
    simple_moving_average = calculate_rolling_mean(prices, window_size)
    return simple_moving_average

# Indicator 2 - Bollinger band
def calculate_bollinger_bands(values, window_size=10, num_std_dev=2):
    rolling_mean = calculate_rolling_mean(values, window_size)
    rolling_std = calculate_rolling_std(values, window_size)
    upper_band = rolling_mean + rolling_std * num_std_dev
    lower_band = rolling_mean - rolling_std * num_std_dev
    return upper_band, lower_band

#Indicator 3 - Momentum 
def calculate_momentum(values, window_size = 10):
    return (values / values.shift(window_size)) - 1
    
# Indicator 4 - Commodity Channel Index - CCI 
def calculate_commodity_channel_index(values, window_size=10):
    rolling_mean = calculate_rolling_mean(values, window_size)
    rolling_std = calculate_rolling_std(values, window_size)
    scaling_factor = 2 / rolling_std
    return (values - rolling_mean) / (scaling_factor * rolling_std)

# Indicator 5 - Moving Average Convergence Divergence (MACD)
def calculate_moving_average_convergence_divergence(values, short_period = 12, long_period = 26, signal_period = 9):    
    # Calculate the short-term EMA
    short_ema = values.ewm(ignore_na=False, span=short_period, adjust=True).mean()
    # Calculate the long-term EMA
    long_ema = values.ewm(ignore_na=False, span=long_period, adjust=True).mean()
    # Calculate the MACD line
    macd_line = short_ema - long_ema
    # Calculate the signal line (9-period EMA of MACD)
    signal_line = macd_line.ewm(ignore_na=False, span=signal_period, adjust=True).mean()
    return macd_line, signal_line
    
def test_code():
    start_date = dt.datetime(2022, 10, 25)
    end_date = dt.datetime(2022, 10, 23)
    symbol_1 = "HSY"
    symbol_2 = "MDLZ"
    symbol_3 = "NSRGY"

    prices_df_HSY = get_data([symbol_1], pd.date_range(start_date, end_date))[symbol_1]
    normalized_prices_df_HSY = prices_df_HSY/prices_df_HSY[0]

    prices_df_HSY = get_data([symbol_2], pd.date_range(start_date, end_date))[symbol_2]
    normalized_prices_df_MDLZ = prices_df_HSY/prices_df_HSY[0]

    prices_df_HSY = get_data([symbol_3], pd.date_range(start_date, end_date))[symbol_3]
    normalized_prices_df_NSRGY = prices_df_HSY/prices_df_HSY[0]
    
        # Indicator 1 Graph
    sma_HSY = calculate_rolling_mean(normalized_prices_df_HSY, 20)
    sma_MDLZ = calculate_rolling_mean(normalized_prices_df_MDLZ, 20)
    sma_NSRGY = calculate_rolling_mean(normalized_prices_df_NSRGY, 20)

    fig = plt.figure(figsize=(10, 6))
    plt.title("Simple Moving average")
    plt.ylabel("Normalized Price")
    plt.xlabel("Date")
    plt.plot(sma_HSY, label="HSY", color="purple", lw=0.8)
    plt.plot(sma_MDLZ, label="MDLZ", color="red", lw=0.8) 
    plt.plot(sma_NSRGY, label="NSRGY", color="blue", lw=0.8) 
    plt.grid(which='both', axis='both')
    plt.tick_params(axis='x', which='major', labelsize=10)
    fig.autofmt_xdate()
    plt.legend(loc="best", frameon=True)
    plt.savefig("images/sma_all.png")
    plt.clf()
    
    # Indicator 2 Graphs
    upper_band, lower_band = calculate_bollinger_bands(normalized_prices_df_HSY,20)
    bollinger_band_percentage = (normalized_prices_df_HSY - lower_band) / (upper_band - lower_band)
    fig = plt.figure(figsize=(10, 5))
    plt.title("HSY Bollinger Bands")
    plt.xticks(rotation=20)
    plt.ylabel("Normalized Values")
    plt.xlabel("Date")
    plt.plot(normalized_prices_df_HSY, label="HSY prices", color="blue", lw=0.8)
    plt.plot(upper_band, label="Bollinger Band Upper Band", color="purple", lw=0.8)
    plt.plot(lower_band, label="Bollinger Band Lower Band", color="red", lw=0.8)
    plt.fill_between(normalized_prices_df_HSY.index, lower_band, upper_band, alpha=0.2, color='red')
    plt.grid(which='both', axis='both', linestyle="--")
    plt.tick_params(axis='x', which='major', labelsize=10)
    fig.autofmt_xdate()
    plt.legend(loc="best", frameon=True)
    plt.savefig("images/HSY_bollinger.png")
    plt.clf()

    upper_band, lower_band = calculate_bollinger_bands(normalized_prices_df_MDLZ,20)
    bollinger_band_percentage = (normalized_prices_df_MDLZ - lower_band) / (upper_band - lower_band)
    fig = plt.figure(figsize=(10, 5))
    plt.title("MDLZ Bollinger Bands")
    plt.xticks(rotation=20)
    plt.ylabel("Normalized Values")
    plt.xlabel("Date")
    plt.plot(normalized_prices_df_MDLZ, label="MDLZ prices", color="blue", lw=0.8)
    plt.plot(upper_band, label="Bollinger Band Upper Band", color="purple", lw=0.8)
    plt.plot(lower_band, label="Bollinger Band Lower Band", color="red", lw=0.8)
    plt.fill_between(normalized_prices_df_MDLZ.index, lower_band, upper_band, alpha=0.2, color='red')
    plt.grid(which='both', axis='both', linestyle="--")
    plt.tick_params(axis='x', which='major', labelsize=10)
    fig.autofmt_xdate()
    plt.legend(loc="best", frameon=True)
    plt.savefig("images/MDLZ_bollinger.png")
    plt.clf()

    upper_band, lower_band = calculate_bollinger_bands(normalized_prices_df_NSRGY,20)
    bollinger_band_percentage = (normalized_prices_df_NSRGY - lower_band) / (upper_band - lower_band)
    fig = plt.figure(figsize=(10, 5))
    plt.title("NSRGY Bollinger Bands")
    plt.xticks(rotation=20)
    plt.ylabel("Normalized Values")
    plt.xlabel("Date")
    plt.plot(normalized_prices_df_NSRGY, label="NSRGY prices", color="blue", lw=0.8)
    plt.plot(upper_band, label="Bollinger Band Upper Band", color="purple", lw=0.8)
    plt.plot(lower_band, label="Bollinger Band Lower Band", color="red", lw=0.8)
    plt.fill_between(normalized_prices_df_NSRGY.index, lower_band, upper_band, alpha=0.2, color='red')
    plt.grid(which='both', axis='both', linestyle="--")
    plt.tick_params(axis='x', which='major', labelsize=10)
    fig.autofmt_xdate()
    plt.legend(loc="best", frameon=True)
    plt.savefig("images/NSRGY_bollinger.png")
    plt.clf()
    
    # Indicator 3 Graph
    momentum_HSY = calculate_momentum(normalized_prices_df_HSY, 20)
    momentum_MDLZ = calculate_momentum(normalized_prices_df_MDLZ, 20)
    momentum_NSRGY = calculate_momentum(normalized_prices_df_NSRGY, 20)

    fig = plt.figure(figsize=(10, 6))
    plt.title("Momentum")
    plt.ylabel("Momentum")
    plt.xlabel("Date")
    plt.plot(momentum_HSY, label="HSY Momentum", color="purple", lw=0.8)
    plt.plot(momentum_MDLZ, label="MDLZ Momentum", color="red", lw=0.8)
    plt.plot(momentum_NSRGY, label="NSRGY Momentum", color="blue", lw=0.8)

    plt.grid(which='both', axis='both', linestyle="--")
    plt.tick_params(axis='x', which='major', labelsize=10)
    plt.axhline(y=0, color="red", lw=0.5)
    plt.legend(loc="best", frameon=True)
    fig.autofmt_xdate()
    plt.savefig("images/momentum_all.png")
    plt.clf()
    
    # Indicator 4 Graph
    cci_HSY = calculate_commodity_channel_index(normalized_prices_df_HSY)
    cci_MDLZ = calculate_commodity_channel_index(normalized_prices_df_MDLZ)
    cci_NSRGY = calculate_commodity_channel_index(normalized_prices_df_NSRGY)

    fig = plt.figure(figsize=(10, 6))
    plt.title("git st")
    plt.ylabel("CCI")
    plt.xlabel("Date")
    plt.plot(cci_HSY, label="HSY CCI", color="purple", lw=0.8)
    plt.plot(cci_MDLZ, label="MDLZ CCI", color="red", lw=0.8)
    plt.plot(cci_NSRGY, label="NSRGY CCI", color="blue", lw=0.8)

    plt.grid(which='both', axis='both', linestyle="--")
    plt.tick_params(axis='x', which='major', labelsize=10)
    plt.axhline(y=0, color="red", lw=0.5)
    plt.legend(loc="best", frameon=True)
    fig.autofmt_xdate()
    plt.savefig("images/cci_all.png")
    plt.clf()
    
    # Indicator 5 Graphs
    macd_line, signal_line = calculate_moving_average_convergence_divergence(normalized_prices_df_HSY)
    fig = plt.figure(figsize=(10, 6))
    plt.suptitle("HSY Moving Average Convergence Divergence")
    ax1 = plt.subplot(211)
    plt.grid(which='both', axis='both', linestyle="--")
    plt.tick_params(axis='x', which='major', labelsize=10)
    plt.ylabel("MACD")
    plt.xlabel("Date")
    ax1.plot(macd_line, label="HSY MACD", color="purple", lw=0.8)
    ax1.plot(signal_line, label="HSY Signal Line", color="blue", lw=0.8)
    plt.legend(loc="best", frameon=True)
    
    short_ema = normalized_prices_df_HSY.ewm(ignore_na=False, span=12, adjust=True).mean()
    long_ema = normalized_prices_df_HSY.ewm(ignore_na=False, span=26, adjust=True).mean()
    ax2 = plt.subplot(212)
    plt.ylabel("EMA")
    plt.xlabel("Date")
    plt.tick_params(axis='x', which='major', labelsize=10)
    plt.grid(which='both', axis='both', linestyle="--")
    plt.plot(normalized_prices_df_HSY, label="HSY Prices", color="blue", lw=0.8)
    ax2.plot(short_ema, label="HSY Short EMA", color="purple", lw=0.8)
    ax2.plot(long_ema, label="HSY Long EMA", color="orange", lw=0.8)
    plt.legend(loc="best", frameon=True)
    fig.autofmt_xdate()
    plt.savefig("images/HSY_macd.png")
    plt.clf()

    macd_line, signal_line = calculate_moving_average_convergence_divergence(normalized_prices_df_MDLZ)
    fig = plt.figure(figsize=(10, 6))
    plt.suptitle("MDLZ Moving Average Convergence Divergence")
    ax1 = plt.subplot(211)
    plt.grid(which='both', axis='both', linestyle="--")
    plt.tick_params(axis='x', which='major', labelsize=10)
    plt.ylabel("MACD")
    plt.xlabel("Date")
    ax1.plot(macd_line, label="MDLZ MACD", color="purple", lw=0.8)
    ax1.plot(signal_line, label="MDLZ Signal Line", color="blue", lw=0.8)
    plt.legend(loc="best", frameon=True)
    
    short_ema = normalized_prices_df_MDLZ.ewm(ignore_na=False, span=12, adjust=True).mean()
    long_ema = normalized_prices_df_MDLZ.ewm(ignore_na=False, span=26, adjust=True).mean()
    ax2 = plt.subplot(212)
    plt.ylabel("EMA")
    plt.xlabel("Date")
    plt.tick_params(axis='x', which='major', labelsize=10)
    plt.grid(which='both', axis='both', linestyle="--")
    plt.plot(normalized_prices_df_MDLZ, label="HSY Prices", color="blue", lw=0.8)
    ax2.plot(short_ema, label="MDLZ Short EMA", color="purple", lw=0.8)
    ax2.plot(long_ema, label="MDLZ Long EMA", color="orange", lw=0.8)
    plt.legend(loc="best", frameon=True)
    fig.autofmt_xdate()
    plt.savefig("images/MDLZ_macd.png")
    plt.clf()

    macd_line, signal_line = calculate_moving_average_convergence_divergence(normalized_prices_df_NSRGY)
    fig = plt.figure(figsize=(10, 6))
    plt.suptitle("NSRGY Moving Average Convergence Divergence")
    ax1 = plt.subplot(211)
    plt.grid(which='both', axis='both', linestyle="--")
    plt.tick_params(axis='x', which='major', labelsize=10)
    plt.ylabel("MACD")
    plt.xlabel("Date")
    ax1.plot(macd_line, label="NSRGY MACD", color="purple", lw=0.8)
    ax1.plot(signal_line, label="NSRGY Signal Line", color="blue", lw=0.8)
    plt.legend(loc="best", frameon=True)
    
    short_ema = normalized_prices_df_HSY.ewm(ignore_na=False, span=12, adjust=True).mean()
    long_ema = normalized_prices_df_HSY.ewm(ignore_na=False, span=26, adjust=True).mean()
    ax2 = plt.subplot(212)
    plt.ylabel("EMA")
    plt.xlabel("Date")
    plt.tick_params(axis='x', which='major', labelsize=10)
    plt.grid(which='both', axis='both', linestyle="--")
    plt.plot(normalized_prices_df_HSY, label="HSY Prices", color="blue", lw=0.8)
    ax2.plot(short_ema, label="NSRGY Short EMA", color="purple", lw=0.8)
    ax2.plot(long_ema, label="NSRGY Long EMA", color="orange", lw=0.8)
    plt.legend(loc="best", frameon=True)
    fig.autofmt_xdate()
    plt.savefig("images/NSRGY_macd.png")
    plt.clf()

# Indicator 6
    fig = plt.figure(figsize=(10, 6))
    plt.title("Normalized Prices")
    plt.ylabel("Prices")
    plt.xlabel("Date")
    plt.plot(normalized_prices_df_HSY, label="HSY Prices", color="red", lw=0.8)
    plt.plot(normalized_prices_df_MDLZ, label="MDLZ Prices", color="green", lw=0.8)
    plt.plot(normalized_prices_df_NSRGY, label="NSRGY Prices", color="blue", lw=0.8)
    plt.grid(which='both', axis='both', linestyle="--")
    plt.tick_params(axis='x', which='major', labelsize=10)
    plt.axhline(y=0, color="red", lw=0.5)
    plt.legend(loc="best", frameon=True)
    fig.autofmt_xdate()
    plt.savefig("images/prices_all.png")
    plt.clf()
    

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)  # Display all rows
    pd.set_option('display.max_columns', None)  # Display all columns
    test_code()