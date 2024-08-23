# OHLC Trendlines Finder

Finds current trendlines based on OHLC (Open, High, Low, Close) data. Given an OHLC dataset, the program determines all relevant trendlines at the closing price of the most recent candle.

 ![Alt text](https://www.dropbox.com/scl/fi/fe459xkw4chq0d4y5ru2t/Screenshot-2024-08-23-at-20.49.17.jpg?rlkey=dt06439zgafp11ye5ficbip61&st=urzurwru&dl=0)


## Features
- **Input:** OHLC dataset containing time series of financial data.
- **Output:** List of active trendlines as of the most recent candle’s close.

## How It Works
1. **Data Input:** Takes an OHLC dataset with columns for open time, timestamp, open, high, low, close, volume, candle type, and candle ID.
2. **Trendline Identification:** Analyzes the dataset to find and record all current trendlines.
3. **Output:** Provides the trendlines valid at the close of the latest candle, including information on their start and end points, time range, and price range.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/trendline-finder.git
