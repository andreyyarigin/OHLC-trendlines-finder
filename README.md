# OHLC Trendlines Finder
It is quite common to see stock market charts with "trendlines" that are not actually trendlines. For novice traders, who often perceive various patterns on the charts (such as "diamond", "head-and-shoulders", "teacup" and other patterns that might eventually play out), it is easy to connect some extreme points on the chart and label it as a "trendline".

In this project, I explain what **Correct Trendlines** are and provide Python code for identifying correct trendlines in OHLC data.

![Screenshot 2024-08-23 at 20 49 17](https://github.com/user-attachments/assets/aad153fb-ca31-436a-8be0-e02163bb6f75)

## Rules for identifying true trends and constructing correct trendlines

### Constructing Trend Intervals

A *Trend* is a tendency of directional change. 
In the context of describing price movement using OHLC data, a trend is defined by a *Trend interval*.

A trend interval is the segment between two adjacent *Global Extrema*.  Global extrema are the absolute maximum and minimum within the considered segment. The segment between them is the trend interval. The direction of the trend interval is determined by which extremum comes first: maximum (MAX) or minimum (MIN).

If the first global extremum is MAX, then the first trend interval is a downtrend interval. It continues until the absolute minimum (MIN). The next trend interval is an uptrend interval and lasts until the following global maximum (MAX) on the remaining segment.

Historical data will thus be represented as a sequence of alternating trend intervals separated by consecutive global extrema.

The process of constructing sequential trend intervals continues until it is no longer possible to construct trendlines on the next remaining segment.

Uptrend/downtrend lines are rays constructed from the minima/maxima of candles within the corresponding trend intervals.

### Constructing Trendlines

A trendline is a ray constructed from two HIGHs of candles (for a downtrend line) or two LOWs of candles (for an uptrend line). 
The construction process follows these conditions (for UPtrendline):
* The first candle is an extremum (MIN).
* The second candle is the one immediately following the first.
* The LOW of the second candle is higher than the LOW of the first candle.
* On the segment following the second candle, the HIGH of the second candle was exceeded (updated) before the LOW of the first candle was updated.

If these conditions are met, the *uptrendline can be drawn from the LOWs of candles on this segment such that no other candles intersect between the first and second points*.

The minimum number of candles required to determine a trendline is 3 (provided that 3[high] > 2[high]).

The second point of the previous trendline is used as the starting point for the next trendline in the same direction.

The construction of trendlines within a trend interval continues as long as the conditions for drawing trendlines are met.

A trendline, where the first point lies on a global extremum, continues until it intersects with another trendline or is no longer valid according to the specified conditions.

A trendline with its first point on a global extremum is called an external trendline. Subsequent trendlines within the same trend interval are called internal trendlines.


