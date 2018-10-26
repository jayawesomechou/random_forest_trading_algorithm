# random_forest_trading_algorithm

This is a trading algorithm using Random Forest as the prediction model to predict the price change of ETF of SPY, ETN of VIX(or the inverse short ETN XIV), 


The performance of this strategy has a 85-90% return in the year of 2016 and 17-20% return in 2017.
* On Feb 5th,2018, there's a flash crash on the US stock market which causes a high volatility rise and that led to a increase in the volitality. The algorithm made the right prediction and get a 40% return in this one single day.


The preidction model takes a historical range of 200 days and look back period of 5 days. The data used is the daily difference of the price and trading volume of the ETF of SPY under the assumption that the trading statistics of SPY ETF represents the market statistics well.







