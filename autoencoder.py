#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 16:12:35 2017

@author: phil
"""



#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.models import load_model

#%%
'''
Load All Data Here

for train_data, test_data
    83 row: stock
    104 column: date   
'''
## load ibb all 3 frames
ibb = pd.read_csv('full_data/ibb_uq.csv', header=0).T # date index: 0 ~ 225
ibb_lp = ibb.iloc[1,:].astype('float32') # all last price
ibb_lp_train = ibb_lp.iloc[0:104]
ibb_lp_test = ibb_lp.iloc[104:208]

ibb_net = ibb.iloc[2,:].astype('float32') # all net change
ibb_net_train = ibb_net.iloc[0:104]
ibb_net_test = ibb_net.iloc[104:208]


## load stock last price
stock_lp = pd.read_csv('full_data/last_price.csv', header=0).T.drop(["Unnamed: 0"]).dropna(axis=0, how='any').astype('float32') # inlcude date
stock_lp_train = stock_lp.iloc[0:, 0:104] # 2012.01 ~ 2013.12
stock_lp_test = stock_lp.iloc[0:, 104:208] # 2014.01 ~ 2016.04


## load stock net change
print("Loading the data ...")
stock_net = pd.read_csv('full_data/net_change.csv', header=0).T.drop(["Unnamed: 0"]).dropna(axis=0, how='any').astype('float32')  # include date,  # axis=0: x-axis
stock_net_train = stock_net.iloc[0:, 0:104] # 2012.01 ~ 2013.12
stock_net_test = stock_net.iloc[0:, 104:208] # 2014.01 ~ 2016.04


## global variable
encoding_dim = 5 # 5 neurons
#%%
'''
AMGN: smallest 2-norm diff -> should be the highest communal stock
BCRX: largest 2-norm diff -> should be the lowest commnual stock
10 most communal stocks + x most non-communal stocks
'''

# predict result
autoencoder = load_model('autoencoder.h5')
decoded_net_train = autoencoder.predict(stock_net_train.as_matrix())

communal_information = []

for i in range(0,83):
    difference = np.linalg.norm((stock_net_train.iloc[i,:]-decoded_net_train[i,:])) # 2 norm difference
    communal_information.append(float(difference))
 

stock_to_rank = np.array(communal_information).argsort()
for row_number in stock_to_rank:
    print(row_number, communal_information[row_number], stock_net_train.iloc[row_number,:].name) #print stock name from lowest to highest

#%%

'''
View highest commnual stock (68 1.3588824272155762 TLGT    US Equity) last_price by:
    
    1. load last_price data
    2. plot original last_price
    3. plot predict last_price
    
Also lowest communal stock (4 92.86773681640625 REGN    US Equity)    

for 1 chosen stock, plot its original price & autoencoded price for comparison
'''

which_stock = 1
# original last price plot
pd.Series(stock_lp_train.iloc[which_stock,0:].as_matrix(), index=pd.date_range(start='01/06/2012', periods=104, freq='W')).plot(label='stock original', legend=True)

# now decoded last price plot
stock_autoencoder = []
price = 0
for i in range(0,104):
    if i == 0:
        price = stock_lp_train.iloc[which_stock,0]
    else:
        price = price + decoded_net_train[which_stock,i]
    stock_autoencoder.append(price)

    
pd.Series(stock_autoencoder, index=pd.date_range(start='01/06/2012', periods = 104,freq='W')).plot(label='stock auto', legend=True)
#%%

'''
Phase 2: Calibration 
    1. load ibb last_price (will do ibb modification in next .py)
    2. plot ibb last_price
    3. infer the weight for S25, S45, S65
    4. plot portfolio price
'''

# Infer weight

'''
10 most commnual stock + x most non-communal stock
Here x = 15, 35, 55, 
so S25, S45, S65
'''
x = 15
s = x+10
port_index = np.concatenate((stock_to_rank[0:10], stock_to_rank[-x:])) # portfolio index
port_lp_train = stock_net_train.iloc[port_index, 0:].T # we have S25 here
port_net_train = stock_net_train.iloc[port_index, 0:].T # we have S25 here

deep_learner = load_model('deep_learner.h5')
# predict deep-learned ibb net change
ibb_learner_net_train = deep_learner.predict(port_net_train.as_matrix())

# calculate deep-learned ibb last price
ibb_autoencoder = []
price = 0
for i in range(0,104):
    if i == 0:
        price = ibb_lp[0] # 2012.1.6 last price
    else:
        price = price + ibb_learner_net_train[i,0]
    ibb_autoencoder.append(price)
 
pd.Series(ibb_autoencoder, index=pd.date_range(start='01/06/2012', periods = 104,freq='W')).plot(label='ibb auto', legend=True)
pd.Series(ibb_lp_train.as_matrix(), index=pd.date_range(start='01/06/2012', periods=104, freq='W')).plot(label='ibb origin', legend=True)    
#%%

'''
Validation Phase
    1. load ibb last_price (will do ibb modification in next .py)
    2. plot ibb last_price
    3. use infered weight to re-track ibb index
    3. plot portfolio price
'''

ibb_plot_test_origin = pd.Series(np.array(ibb_last_test), index=pd.date_range(start='01/06/2013', periods=104, freq='W'))
ax= ibb_plot_train_origin.plot()
ax.set_xlabel('Weekly Date')
ax.set_ylabel('IBB Last Price')


#ibb_learner_test = deep_learner.predict() # plug in S25 test date data

#%%

'''
Verification Phase: Deep Frontier
'''













