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

input_img = Input(shape=(104,))

encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_img)
decoded = Dense(104, activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)
autoencoder = Model(input_img, decoded)

# predict model
encoder = Model(input_img, encoded)# this model maps an input to its encoded representation
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))


autoencoder.compile(loss='mean_squared_error', optimizer='sgd')


# train autoencoder
train_np = train_data.as_matrix()
autoencoder.fit(train_np, train_np, epochs=1000)

#%%

# predict result
encoded_imgs = encoder.predict(train_np)
decoded_imgs = decoder.predict(encoded_imgs)

#%%

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
 
#%%     

'''
AMGN: smallest 2-norm diff -> should be the highest communal stock
BCRX: largest 2-norm diff -> should be the lowest commnual stock


10 most communal stocks + x most non-communal stocks
'''

stock_to_rank = np.array(communal_information).argsort()

for row_number in stock_to_rank:
    print(row_number, communal_information[row_number], train_data.iloc[row_number,:].name) #print stock name from lowest to highest






















