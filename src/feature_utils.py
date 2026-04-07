import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import pandas_datareader.data as web
import requests
#from datetime import datetime, timedelta
import os
import sys
import json #

from src.Custom_Classes import FeatureEngineer


def extract_features():

    return_period = 5
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    file_path = os.path.join(project_root, 'Portfolio/SP500Data.csv')

    dataset = pd.read_csv(file_path, index_col=0)

    target = 'AMZN'

    Y = np.log(dataset[[target]]).diff(return_period).shift(-return_period)
    Y = np.exp(Y).cumsum()
    Y.columns = [f'{target}_FR_Cum']
    
    X = np.log(dataset.drop([target], axis=1)).diff(return_period)
    X = np.exp(X).cumsum()
    X.columns = [name + '_CR_Cum' for name in X.columns]

    dataset = pd.concat([Y, X], axis=1).dropna().iloc[::return_period, :]
    Y = dataset.loc[:, Y.columns[0]]
    X = dataset.loc[:, X.columns]
    dataset.index.name = 'Date'
    features = dataset.sort_index()
    features = features.reset_index(drop=True)
    features = features.iloc[:,1:]
    return features


def extract_features_pair():

    START_DATE = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    END_DATE = datetime.date.today().strftime("%Y-%m-%d")
    stk_tickers = ['AMZN', 'NVDA']
    
    stk_data = yf.download(stk_tickers, start=START_DATE, end=END_DATE, auto_adjust=False)

    Y = stk_data.loc[:, ('Adj Close', 'AMZN')]
    Y.name = 'AMZN'

    X = stk_data.loc[:, ('Adj Close', 'NVDA')]
    X.name = 'NVDA'

    dataset = pd.concat([Y, X], axis=1).dropna()
    Y = dataset.loc[:, Y.name]
    X = dataset.loc[:, X.name]
    dataset.index.name = 'Date'
    features = dataset.sort_index()
    features = features.reset_index(drop=True)
    return features


def get_bitcoin_historical_prices(days = 60):
    
    BASE_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily' # Ensure we get daily granularity
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['Timestamp', 'Close Price (USD)'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.normalize()
    df = df[['Date', 'Close Price (USD)']].set_index('Date')
    return df


def convert_input_pca_regression(request_body, request_content_type):
    print(f"Receiving data of type: {request_content_type}")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    file_path = os.path.join(project_root, 'Portfolio/SP500Data.csv')

    dataset = pd.read_csv(file_path,index_col=0)

    target = 'AMZN'

    option = 1

    if option == 2:

        X = FeatureEngineer(windows=[10,15]).transform(dataset[[target]])
    
        techIndicator_1 = 'RSI_15'
        RSI_15 = json.loads(request_body)[techIndicator_1]
        techIndicator_2 = 'MOM_15'
        MOM_15 = json.loads(request_body)[techIndicator_2]

        # Calculate the distance
        distances = np.sqrt(
            (X[techIndicator_1] - RSI_15)**2 + 
            (X[techIndicator_2] - MOM_15)**2
        )
        
        closest_index = distances.idxmin()
        closest_row = X.loc[[closest_index]]
    
        closest_row[techIndicator_1] = RSI_15
        closest_row[techIndicator_2] = MOM_15
    
        return closest_row
    else:

        return_period = 5

        SP500_1 = 'IBM_CR_Cum'
        IBM_CR_Cum = json.loads(request_body)[SP500_1]
        SP500_2 = 'NVDA_CR_Cum'
        NVDA_CR_Cum = json.loads(request_body)[SP500_2]

        X = np.log(dataset.drop([target],axis=1)).diff(return_period)
        X = np.exp(X).cumsum()
        X.columns = [name + "_CR_Cum" for name in X.columns]
        
        # Calculate the distance
        distances = np.sqrt(
            (X[SP500_1] - IBM_CR_Cum)**2 + 
            (X[SP500_2] - NVDA_CR_Cum)**2
        )
        
        closest_index = distances.idxmin()
        closest_row = X.loc[[closest_index]]
    
        closest_row[SP500_1] = IBM_CR_Cum
        closest_row[SP500_2] = NVDA_CR_Cum
    
        return closest_row
