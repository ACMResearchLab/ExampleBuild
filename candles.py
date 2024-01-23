import pandas as pd
from dotenv import dotenv_values
import polygon
import datetime as dt
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import os


# getBarData() | takes in a stock ticker and returns the last 28 daily bars
def getBarData(stock):
    config = dotenv_values(".env") #api key and other things

    #Gets the dates for a little over three years ago from today
    start = (dt.datetime.today() - timedelta(days=1500)).date().isoformat()
    end = dt.datetime.today().date().isoformat()

    # create an api instance and get the data
    api = polygon.StocksClient(config.get("API_KEY"))
    data = pd.DataFrame(api.get_aggregate_bars(stock, start, end)['results'])

    # Rename columns for the prewritten functions
    data = data.rename(columns={'o': 'Open', 'c': 'Close', 'h': 'High', 'l': 'Low', 'v': 'Volume', 't': 'Time'})

    # Convert the timestamp to datetime and set as index
    data['Time'] = pd.to_datetime(data['Time'], unit='ms')
    data = data.set_index('Time')

    return data[['Open', 'Close', 'High', 'Low', 'Volume']]




def plot_candles(pricing, title=None, volume_bars=False, color_function=None, technicals=None):
    """ Plots a candlestick chart using quantopian pricing data.
    
    Author: Daniel Treiman
    
    Args:
      pricing: A pandas dataframe with columns ['open_price', 'close_price', 'high', 'low', 'volume']
      title: An optional title for the chart
      volume_bars: If True, plots volume bars
      color_function: A function which, given a row index and price series, returns a candle color.
      technicals: A list of additional data series to add to the chart.  Must be the same length as pricing.
    """
    def default_color(index, open_price, close_price, low, high):
        return 'r' if open_price.iloc[index] > close_price.iloc[index] else 'g'
    color_function = color_function or default_color
    technicals = technicals or []
    open_price = pricing['Open']
    close_price = pricing['Close']
    low = pricing['Low']
    high = pricing['High']
    oc_min = pd.concat([open_price, close_price], axis=1).min(axis=1)
    oc_max = pd.concat([open_price, close_price], axis=1).max(axis=1)
    
    if volume_bars:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3,1]},figsize=(7,7))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(3,2))
    if title:
        ax1.set_title(title)
    fig.tight_layout()
    x = np.arange(len(pricing))
    candle_colors = [color_function(i, open_price, close_price, low, high) for i in x]
    ax1.bar(x, oc_max-oc_min, bottom=oc_min, color=candle_colors, linewidth=0)
    ax1.vlines(x , low, high, color=candle_colors, linewidth=1)
    ax1.axis('off')
    plt.margins(0,0)
    ax1.xaxis.grid(False)
    ax1.yaxis.grid(False)
    ax1.xaxis.set_tick_params(which='major', length=3.0, direction='in', top='off')
    ax1.set_yticklabels([])
    # Assume minute frequency if first two bars are in the same day.
    frequency = 'minute' if (pricing.index[1] - pricing.index[0]).days == 0 else 'day'
    time_format = '%d-%m-%Y'
    if frequency == 'minute':
        time_format = '%H:%M'
    # Set X axis tick labels.
    #plt.xticks(x, [date.strftime(time_format) for date in pricing.index], rotation='vertical')
    for indicator in technicals:
        ax1.plot(x, indicator)
    
    if volume_bars:
        volume = pricing['Volume']
        volume_scale = None
        scaled_volume = volume
        if volume.max() > 1000000:
            volume_scale = 'M'
            scaled_volume = volume / 1000000
        elif volume.max() > 1000:
            volume_scale = 'K'
            scaled_volume = volume / 1000
        ax2.bar(x, scaled_volume, color=candle_colors)
        volume_title = 'Volume'
        if volume_scale:
            volume_title = 'Volume (%s)' % volume_scale
        #ax2.set_title(volume_title)
        ax2.xaxis.grid(False)
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        

    return fig    


def price_trend(stocks, start_date, end_date, n_days=5, fraction_movement=0.037):
    for stock in stocks:
        print(f"Analyzing stock: {stock}")
        df_pricing = getBarData(stock)  # Assuming getBarData fetches data between start_date and end_date
        df = df_pricing.copy()
        df = df.reset_index()

        df['Trend'] = None
        for i in range(len(df)):
            try:
                for n in range(1, n_days + 1):  # Start from 1 to avoid comparing the day with itself
                    if i + n < len(df):  # Check to avoid index out of range
                        if df.loc[i, 'Close'] - df.loc[i + n, 'Close'] >= fraction_movement * df.loc[i, 'Close']:
                            df.loc[i, 'Trend'] = 'Down'
                            if i >= 20:
                                fig = plot_candles(df_pricing.iloc[i - 20:i + 1], volume_bars=False)
                                file_path = os.path.join('Candle Data', 'Down', f'{stock}{i}.png')
                                plt.savefig(file_path, format='png', dpi=70)
                                print(f"Down trend plot saved to {file_path}")
                            break
                        elif df.loc[i + n, 'Close'] - df.loc[i, 'Close'] >= fraction_movement * df.loc[i, 'Close']:
                            df.loc[i, 'Trend'] = 'Up'
                            if i >= 20:
                                fig = plot_candles(df_pricing.iloc[i - 20:i + 1], volume_bars=False)
                                file_path = os.path.join('Candle Data', 'Up', f'{stock}{i}.png')
                                plt.savefig(file_path, format='png', dpi=50)
                                print(f"Up trend plot saved to {file_path}")
                            break
            except Exception as e:
                print(f"Error at index {i} for stock {stock}: {e}")


def updatedCandles(stocks):
    image_count = 1
    for stock in stocks:
        df_pricing = getBarData(stock)
        df = df_pricing.copy()
        df = df.reset_index(drop=True)
        n_days = 5
        df_len = len(df)
        file_path = os.path.join('Candle Data', 'Training Data/')
        for i in range(len(df)):
            try:
                for n in range(n_days):
                    if i >= 28:
                        fig=plot_candles(df_pricing[i-28:i],volume_bars=False)
                        fig.savefig(f'{file_path}{image_count},{df_pricing["Close"].iloc[i]},{stock}.png')
                        plt.close(fig)
                        image_count += 1
                        print('Down',i,n)
                    break
                print("The date is: ", df_pricing.index())
            except:
                pass

# # #roughly two years ago start date to todays date end date
# # start = (dt.datetime.today() - timedelta(days=1500)).date().isoformat()
# # end = dt.datetime.today().date().isoformat()
# # #apple_data = getBarData("AAPL")
# stocks = ['AAPL', 'MSFT', 'NVDA', 'AMD', 'V', 'ADBE', 'AMZN', 'INTC', 'EA', 'NFLX', 'QCOM', 'TSLA', 'MDB']
# # #price_trend(stocks, start, end)
# updatedCandles(stocks)