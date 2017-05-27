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

# predict model
encoder = Model(input_img, encoded)# this model maps an input to its encoded representation
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))


autoencoder.compile(loss='mean_squared_error', optimizer='sgd')
#%%

# train autoencoder
train_np = train_data.as_matrix()
autoencoder.fit(train_np, train_np, shuffle=True, epochs=5000)


'''
encoded_imgs = encoder.predict(train_np)
decoded_imgs = decoder.predict(encoded_imgs)
'''
#%%
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
which_stock = 22

stock_origin = pd.Series(train_last_price.iloc[which_stock,0:].as_matrix(), index=pd.date_range(start='01/06/2012', periods=104, freq='W'))
stock_origin.plot()

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
stock_autoencoder.plot()
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
ibb_plot_train_origin.plot()

#%%






#%%

'''
Validation Phase
    1. load ibb last_price (will do ibb modification in next .py)
    2. plot ibb last_price
    3. use infered weight to re-track ibb index
    3. plot portfolio price
'''

#%%

'''
Verification Phase: Deep Frontier
'''













