# https://www.quantopian.com/posts/simple-machine-learning-example-mk-ii
# Use the previous bars' movements to predict the next movement.

# Use a random forest classifier. More here: http://scikit-learn.org/stable/user_guide.html
from sklearn.ensemble import RandomForestRegressor


from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import CustomFactor

# The sample version available from 15 Oct 2012 - 11 Jan 2016
# from quantopian.pipeline.data.sentdex import sentiment_free as sentdex
# The premium version found at https://www.quantopian.com/data/sentdex/sentiment
# from quantopian.pipeline.data.sentdex import sentiment as sentdex

import pandas as pd
import numpy as np


# Calculates the average impact of the sentiment over the window length
# class AvgSentiment(CustomFactor):

#     def compute(self, today, assets, out, impact):
#         np.mean(impact, axis=0, out=out)


# class AvgDailyDollarVolumeTraded(CustomFactor):

#     inputs = [USEquityPricing.close, USEquityPricing.volume]
#     window_length = 20

#     def compute(self, today, assets, out, close_price, volume):
#         out[:] = np.mean(close_price * volume, axis=0)

def initialize(context):
    context.spy = sid(8554) # ETF predictor spy
    context.xiv = sid(40516) # ETF target xiv
    context.vix = sid(40670) # ETF target
    context.xiv_model = RandomForestRegressor()
    context.spy_model = RandomForestRegressor()
    set_benchmark(sid(8554))
    # set_benchmark(sid(40516))
    
    context.lookback = 5 # Look back 5 periods
    context.history_range = 200 # Only consider the past 200 periods' history
    # tree number used
    context.treeNum = 10
    
    context.xiv_pred=0
    context.spy_pred=0
    # Add Sentiment Pipline
    window_length = 5
    # pipe = Pipeline()
    # pipe = attach_pipeline(pipe, name='sentiment_metrics')
    # dollar_volume = AvgDailyDollarVolumeTraded()
    
    # Add our AvgSentiment factor to the pipeline using a 3 day moving average
    # pipe.add(AvgSentiment(inputs=[sentdex.sentiment_signal], window_length=window_length), "avg_sentiment")
    
    # Screen out low liquidity securities.
    context.shorts = None
    context.longs = None
    
    context.count_long=0
    context.count_short=0
    context.count_clear=0
    context.position=0
    context.positionX=0
    
    # context.predict_vol=0
    
    # Generate new models every day
    # schedule_function(create_vol_model, date_rules.every_day(), time_rules.market_close(minutes=10))
    # schedule_function(create_price_model, date_rules.every_day(), time_rules.market_close(minutes=15))
    
    # Make prediction every day
    schedule_function(mult_pred, date_rules.every_day(), time_rules.market_open(minutes=2))
    
    
    # Trade XIV and SPY base on the predictions
    schedule_function(trade, date_rules.every_day(), time_rules.market_open(minutes=5))


# schedule_function(clear_position, date_rules.every_day(), time_rules.market_open(minutes=120))



def before_trading_start(context, data):
    context.position=0
    context.positionX=0
    context.xiv_pred=0
    context.vol_pred=0
    context.valuebeginning=context.portfolio.portfolio_value
# results = pipeline_output('sentiment_metrics').dropna()

# # Separate securities into longs and shorts
# # longs = results[results['avg_sentiment'] > 0]
# # shorts = results[results['avg_sentiment'] < 0]
# longs = results[results['avg_sentiment'] > 0]['avg_sentiment'].sum()
# shorts = results[results['avg_sentiment'] < 0]['avg_sentiment'].sum()

# if longs+shorts>0:
#     context.sentdex=1
# else:
#     context.sentdex=-1


def clear_position(context,data):
    if context.portfolio.portfolio_value<0.98*context.valuebeginning:
        for sid in context.portfolio.positions:
            order_target_percent(sid,0)
        print "sold all stuff"



def mult_pred(context, data):
    for i in range(context.treeNum):
        create_xiv_model(context,data)
        create_spy_model(context,data)
        model_pred(context,data)



def create_xiv_model(context, data):
    # Get the relevant daily volumes
    recent_xiv = data.history(context.spy, 'volume', context.history_range, '1d').values
    
    # Get the volume changes
    xiv_changes = np.diff(recent_xiv).tolist()
    
    X = [] # Independent, or input variables
    Y = [] # Dependent, or output variable
    
    # For each day in our history
    for i in range(context.history_range-context.lookback-1):
        X.append(xiv_changes[i:i+context.lookback]) # Store prior volume changes
        Y.append(xiv_changes[i+context.lookback]) # Store the current volume change
    
    context.xiv_model.fit(X, Y) # Generate our model'
# context.vol_pred=context.vol_model.predict(volume_changes)


def create_spy_model(context, data):
    # Get the relevant daily prices
    recent_spy = data.history(context.spy, 'price', context.history_range, '1d').values
    
    # Get the price changes
    spy_changes = np.diff(recent_spy).tolist()
    
    X = [] # Independent, or input variables
    Y = [] # Dependent, or output variable
    
    # For each day in our history
    for i in range(context.history_range-context.lookback-1):
        X.append(spy_changes[i:i+context.lookback]) # Store prior price changes
        Y.append(spy_changes[i+context.lookback]) # Store the day's price change
    
    context.spy_model.fit(X, Y) # Generate our model
# context.price_pred=context.price_pred+context.price_model.predict(price_changes)

def model_pred(context, data):
    if context.xiv_model and context.spy_model: # Check if our models are generated
        
        # Get recent prices
        recent_xiv = data.history(context.spy, 'volume', context.lookback+1, '1d').values
        recent_spy = data.history(context.spy, 'price', context.lookback+1, '1d').values
        
        # Get the price changes
        xiv_changes = np.diff(recent_xiv).tolist()
        # print volume_changes
        spy_changes = np.diff(recent_spy).tolist()
        
        if(context.spy_model.predict(spy_changes)>0):
            context.position+=1
        else:
            context.position-=1
        
        if(context.xiv_model.predict(xiv_changes)>0):
            context.positionX+=1
        else:
            context.positionX-=1
        # print context.vol_model.predict(volume_changes)
        
        
        context.xiv_pred=context.xiv_pred+context.xiv_model.predict(xiv_changes)
        context.spy_pred=context.spy_pred+context.spy_model.predict(spy_changes)




def trade(context, data):
    
    #context.vol_pred=context.vol_model.predict(volume_changes)
    #context.price_pred=context.price_model.predict(price_changes)
    # Predict using our models and the recent prices and volumes
    
    xiv_prediction = context.xiv_pred / (context.treeNum*1.0)
        record(xiv_prediction = xiv_prediction)
        spy_prediction = context.spy_pred / (context.treeNum*1.0)
        record(spy_prediction = spy_prediction)
        
        
        # Go long if we predict the price and volume will increase
        # Sell if we predict the price and volume will decrease
        # Close orders otherwise
        # Trade XIV instead of SPY, but use SPY model
        # if vol_prediction > 0 and price_prediction > 0 and context.sentdex>0:
        # elif vol_prediction < 0 and price_prediction < 0 and context.sentdex<0:
        context.position=context.position/(context.treeNum*1.0)
        context.positionX=context.positionX/(context.treeNum*1.0)
        order_target_percent(context.spy, context.position/2)
        order_target_percent(context.xiv, context.positionX/2)
        
        if context.position*spy_prediction<0:
            order_target_percent(context.spy,0)
            print "clear SPY"
    if context.positionX*xiv_prediction<0:
        order_target_percent(context.xiv,0)
            print "clear XIV"
        
        if context.position>0:
            print "long SPY", context.position/2
else:
    print "short SPY", context.position/2
        
        if context.positionX>0:
            print "long XIV", context.positionX/2
    else:
        print "short XIV", context.positionX/2






