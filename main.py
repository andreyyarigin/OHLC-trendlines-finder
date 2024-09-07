import numpy as np
import ast
import pandas as pd
import datetime
import time

from functions import *

df = pd.read_csv('/Users/andreyyarigin/projects/OHLC-trendlines-finder/XRPUSDT_OHLCV_1D.csv', header = None, names = ('timestamp', 'open', 'high', 'low', 'close', 'volume'))

df['open_time'] = pd.to_datetime(df['timestamp'], unit='ms')
df['candle_type'] = df.apply(lambda x: 1 if x['open'] <= x['close'] else 0, axis = 1)
df.reset_index(inplace = True, names = 'candle_id')
new_columns_order = ['open_time', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'candle_type', 'candle_id']
otohlcvcc_df = df[new_columns_order]

tohlcvcc_array = convert_df_to_array (otohlcvcc_df)

trend_intervals = find_trend_intervals (tohlcvcc_array)

trendlines = find_all_trendlines (otohlcvcc_df)

print (trendlines)