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


## load data
print("Loading the data ...")
full_data = pd.read_csv('full_data/net_change.csv', header=0).T # inlcude date

data = full_data.drop(["Unnamed: 0"]).dropna(axis=0, how='any').astype('float32') # axis=0: x-axis

train_data = data.iloc[0:, 0:104] # 2012.01 ~ 2013.12
test_data = data.iloc[0:, 104:226] # 2014.01 ~ 2016.04


'''
for train_data
    83 row: stock
    104 column: date
    
for test_data
    83 row: stock
    122 column: date    
'''

encoding_dim = 5 # 5 neurons

# autoencoder model
input_img = Input(shape=(104,))
encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_img)
decoded = Dense(104, activation='linear', activity_regularizer=regularizers.l1(10e-5))(encoded) # see 'Stacked Auto-Encoders' in paper
autoencoder = Model(input_img, decoded)
autoencoder.compile(loss='mean_squared_error', optimizer='sgd')

# train autoencoder
train_np = train_data.as_matrix()
autoencoder.fit(train_np, train_np, shuffle=True, epochs=5000)

# predict result
decoded_imgs = autoencoder.predict(train_np)
#%%
'''
AMGN: smallest 2-norm diff -> should be the highest communal stock
BCRX: largest 2-norm diff -> should be the lowest commnual stock


10 most communal stocks + x most non-communal stocks
'''

# 2 norm difference
communal_information = []

for i in range(0,83):
    '''
    stock_2norm = np.linalg.norm(train_np[i,:])
    auto_2norm = np.linalg.norm(decoded_imgs[i,:])
    communal_information.append(float(abs(stock_2norm - auto_2norm)))
    '''
    difference = np.linalg.norm((train_np[i,:]-decoded_imgs[i,:]))
    communal_information.append(float(difference))
 

stock_to_rank = np.array(communal_information).argsort()
for row_number in stock_to_rank:
    print(row_number, communal_information[row_number], train_data.iloc[row_number,:].name) #print stock name from lowest to highest

#%%

'''
View highest commnual stock (68 1.3588824272155762 TLGT    US Equity) last_price by:
    
    1. load last_price data
    2. plot original last_price
    3. plot predict last_price
    
Also lowest communal stock (4 92.86773681640625 REGN    US Equity)    
'''


full_last_price = pd.read_csv('full_data/last_price.csv', header=0).T # inlcude date

last_price = full_last_price.drop(["Unnamed: 0"]).dropna(axis=0, how='any').astype('float32') # axis=0: x-axis

train_last_price = last_price.iloc[0:, 0:104] # 2012.01 ~ 2013.12

#%%

'''
    for 1 chosen stock, plot its original price & autoencoded price for comparison
'''

which_stock = 76

stock_origin = pd.Series(train_last_price.iloc[which_stock,0:].as_matrix(), index=pd.date_range(start='01/06/2012', periods=104, freq='W'))
stock_origin.plot(label='stock original', legend=True)

# now predict price
stock_autoencoder = []
price = 0
for i in range(0,104):
    if i == 0:
        price = train_last_price.iloc[which_stock,0]
    else:
        price = price + decoded_imgs[which_stock,i]
    stock_autoencoder.append(price)

    
stock_autoencoder = pd.Series(stock_autoencoder, index=pd.date_range(start='01/06/2012', periods = 104,freq='W'))
stock_autoencoder.plot(label='stock autoencoded', legend=True)
#%%

'''
Calibration Phase
    1. load ibb last_price (will do ibb modification in next .py)
    2. plot ibb last_price
    3. infer the weight for S25, S45, S65
    4. plot portfolio price

'''

# Step 1 & 2
full_ibb = pd.read_csv('full_data/ibb_uq.csv', header=0).T # date index: 0 ~ 225
ibb_last_price = full_ibb.iloc[1,:].astype('float') # all last price
ibb_last_train = ibb_last_price.iloc[0:104]
ibb_last_test = ibb_last_price.iloc[104:226]

ibb_plot_train_origin = pd.Series(np.array(ibb_last_train), index=pd.date_range(start='01/06/2012', periods=104, freq='W'))
ax= ibb_plot_train_origin.plot()
ax.set_xlabel('Weekly Date')
ax.set_ylabel('IBB Last Price')

#%%

# step 3: infer weight

'''
10 most commnual stock + x most non-communal stock
Here x = 15, 35, 55, 
so S25, S45, S65
'''
x = 15
s = x+10
most_com_10 = stock_to_rank[0:10]
least_com_10 = stock_to_rank[-x:]
select_index = np.concatenate((most_com_10, least_com_10))
select_stock = train_last_price.iloc[select_index,0:].T # we have S25 here


#%%
# deep learner model
portfolio = Input(shape=(s,)) # row: 104 dates, column: S stocks
learner_hidden = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(portfolio)
learner_output = Dense(1, activation='relu', activity_regularizer=regularizers.l1(10e-5))(learner_hidden) # output layer
deep_learner = Model(portfolio, learner_output)
deep_learner.compile(loss='mean_squared_error', optimizer='sgd')

# train autoencoder
deep_learner.fit(select_stock.as_matrix().astype('float'), ibb_last_train.as_matrix().astype('float'), epochs=5000) # ??? f(stock_last) = ibb_last

# predict result
ibb_learner_train = deep_learner.predict(select_stock.as_matrix())


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













