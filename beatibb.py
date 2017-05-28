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

ibb_per = ibb.iloc[3,:].astype('float64') # all percentage change
ibb_per_train = ibb_per.iloc[0:104]
ibb_per_test = ibb_per.iloc[104:208]


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

which_stock = 27
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

'''
# modify calibration target
smaller_index = [ n for n,i in enumerate(ibb_per_train.as_matrix()) if i<-5 ] # only #92 in this dataset
for i in smaller_index:
    ibb_net_train.iloc[i] = ibb_lp_train.iloc[i-1]*0.05

ibb_modify = []
price = 0

for i in range(0,104):
    if i == 0:
        price = ibb_lp[0] # 2012.1.6 last price
    else:
        price = price + ibb_net_train.as_matrix()[i]
        
    ibb_modify.append(price) 
'''

pd.Series(ibb_lp_train.as_matrix(), index=pd.date_range(start='01/06/2012', periods=104, freq='W')).plot(label='ibb original', legend=True) 
pd.Series(ibb_modify, index=pd.date_range(start='01/06/2012', periods=104, freq='W')).plot(label='ibb modify', legend=True)   
#%%
# train deep learner model for S25, S45, S65
for x in [15,35,55]:
    x = 55
    s = x+10
    port_index = np.concatenate((stock_to_rank[0:10], stock_to_rank[-x:])) # portfolio index
    port_net_train = stock_net_train.iloc[port_index, 0:].T # we have S25 here
    
    portfolio = Input(shape=(s,)) # row: 104 dates, column: 10+x stocks
    learner_hidden = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(portfolio)
    learner_output = Dense(1, activation='linear', activity_regularizer=regularizers.l1(10e-5))(learner_hidden) # output layer
    deep_learner = Model(portfolio, learner_output)
    deep_learner.compile(loss='mean_squared_error', optimizer='sgd')
    deep_learner.fit(port_net_train.as_matrix(), ibb_net_train.as_matrix(), epochs=5000) # ??? f(stock_net) = ibb_net
    
    deep_learner.save(('s'+str(s)+'.h5'))

#%%
s25 = load_model('s25.h5')
s45 = load_model('s45.h5')
s65 = load_model('s65.h5')

x1 = 15
x2 = 35
x3 = 55

port_index_s25 = np.concatenate((stock_to_rank[0:10], stock_to_rank[-x1:])) # portfolio index
port_index_s45 = np.concatenate((stock_to_rank[0:10], stock_to_rank[-x2:])) # portfolio index
port_index_s65 = np.concatenate((stock_to_rank[0:10], stock_to_rank[-x3:])) # portfolio index
                                 
port_net_train_s25 = stock_net_train.iloc[port_index_s25, 0:].T # we have S25 here
port_net_train_s45 = stock_net_train.iloc[port_index_s45, 0:].T # we have S25 here
port_net_train_s65 = stock_net_train.iloc[port_index_s65, 0:].T # we have S25 here

ibb_learner_net_train_s25 = s25.predict(port_net_train_s25.as_matrix()) # predict deep-learned ibb net change
ibb_learner_net_train_s45 = s45.predict(port_net_train_s45.as_matrix()) # predict deep-learned ibb net change
ibb_learner_net_train_s65 = s65.predict(port_net_train_s65.as_matrix()) # predict deep-learned ibb net change

# calculate deep-learned ibb last price
ibb_autoencoder_s25 = []
price_s25 = 0

ibb_autoencoder_s45 = []
price_s45 = 0

ibb_autoencoder_s65 = []
price_s65 = 0

for i in range(0,104):
    if i == 0:
        price_s25 = ibb_lp[0] # 2012.1.6 last price
        price_s45 = ibb_lp[0] # 2012.1.6 last price
        price_s65 = ibb_lp[0] # 2012.1.6 last price
    else:
        price_s25 = price_s25 + ibb_learner_net_train_s25[i,0]
        price_s45 = price_s45 + ibb_learner_net_train_s45[i,0]
        price_s65 = price_s65 + ibb_learner_net_train_s65[i,0]
        
    ibb_autoencoder_s25.append(price_s25)
    ibb_autoencoder_s45.append(price_s45)
    ibb_autoencoder_s65.append(price_s65)
 
pd.Series(ibb_autoencoder_s25, index=pd.date_range(start='01/06/2012', periods = 104,freq='W')).plot(label='ibb S25', legend=True)
pd.Series(ibb_autoencoder_s45, index=pd.date_range(start='01/06/2012', periods = 104,freq='W')).plot(label='ibb S45', legend=True)
pd.Series(ibb_autoencoder_s65, index=pd.date_range(start='01/06/2012', periods = 104,freq='W')).plot(label='ibb S65', legend=True)
pd.Series(ibb_lp_train.as_matrix(), index=pd.date_range(start='01/06/2012', periods=104, freq='W')).plot(label='ibb original', legend=True)  

print("S25 2-norm difference: ", np.linalg.norm((ibb_autoencoder_s25-ibb_lp_train.as_matrix()))) 
print("S45 2-norm difference: ", np.linalg.norm((ibb_autoencoder_s45-ibb_lp_train.as_matrix()))) 
print("S65 2-norm difference: ", np.linalg.norm((ibb_autoencoder_s65-ibb_lp_train.as_matrix()))) 
#%%

'''
Phase 3: Validation 
    1. load ibb last_price (will do ibb modification in next .py)
    2. plot ibb last_price
    3. use infered weight to re-track ibb index
    3. plot portfolio price
'''

# port_index_s25
# port_index_s45
# port_index_s65

port_net_test_s25 = stock_net_test.iloc[port_index_s25, 0:].T # we have S25 here
port_net_test_s45 = stock_net_test.iloc[port_index_s45, 0:].T # we have S25 here
port_net_test_s65 = stock_net_test.iloc[port_index_s65, 0:].T # we have S25 here

# predict deep-learned ibb net change
ibb_learner_net_test_s25 = s25.predict(port_net_test_s25.as_matrix())
ibb_learner_net_test_s45 = s45.predict(port_net_test_s45.as_matrix())
ibb_learner_net_test_s65 = s65.predict(port_net_test_s65.as_matrix())


# calculate deep-learned ibb last price
ibb_autoencoder_s25 = []
price_s25 = 0

ibb_autoencoder_s45 = []
price_s45 = 0

ibb_autoencoder_s65 = []
price_s65 = 0

for i in range(0,104):
    if i == 0:
        price_s25 = ibb_lp[104] # 2012.1.6 last price
        price_s45 = ibb_lp[104] # 2012.1.6 last price
        price_s65 = ibb_lp[104] # 2012.1.6 last price
    else:
        price_s25 = price_s25 + ibb_learner_net_test_s25[i,0]
        price_s45 = price_s45 + ibb_learner_net_test_s45[i,0]
        price_s65 = price_s65 + ibb_learner_net_test_s65[i,0]
        
    ibb_autoencoder_s25.append(price_s25)
    ibb_autoencoder_s45.append(price_s45)
    ibb_autoencoder_s65.append(price_s65)
 
pd.Series(ibb_autoencoder_s25, index=pd.date_range(start='01/03/2014', periods = 104,freq='W')).plot(label='ibb S25', legend=True)
pd.Series(ibb_autoencoder_s45, index=pd.date_range(start='01/03/2014', periods = 104,freq='W')).plot(label='ibb S45', legend=True)
pd.Series(ibb_autoencoder_s65, index=pd.date_range(start='01/03/2014', periods = 104,freq='W')).plot(label='ibb S65', legend=True)
pd.Series(ibb_lp_test.as_matrix(), index=pd.date_range(start='01/03/2014', periods=104, freq='W')).plot(label='ibb original', legend=True)  

print("S25 2-norm difference: ", np.linalg.norm((ibb_autoencoder_s25-ibb_lp_test.as_matrix()))) 
print("S45 2-norm difference: ", np.linalg.norm((ibb_autoencoder_s45-ibb_lp_test.as_matrix()))) 
print("S65 2-norm difference: ", np.linalg.norm((ibb_autoencoder_s65-ibb_lp_test.as_matrix()))) 
#%%

'''
Phase 4: Verification & Deep Frontier

x-axis: 2-norm error
y-axis: # of stocks, 60, 40, 20
'''

error = []
for x in range(5,56,5):

    # 10 commnual + x non-communal
    s = x+10 
    port_index = np.concatenate((stock_to_rank[0:10], stock_to_rank[-x:])) # portfolio index
    port_lp_train = stock_net_train.iloc[port_index, 0:].T # we have S25 here
    port_net_train = stock_net_train.iloc[port_index, 0:].T # we have S25 here
    
    # deep learner model
    portfolio = Input(shape=(s,)) # row: 104 dates, column: 10+x stocks
    learner_hidden = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(portfolio)
    learner_output = Dense(1, activation='linear', activity_regularizer=regularizers.l1(10e-5))(learner_hidden) # output layer
    deep_learner = Model(portfolio, learner_output)
    deep_learner.compile(loss='mean_squared_error', optimizer='sgd')
    deep_learner.fit(port_net_train.as_matrix(), ibb_net_train.as_matrix(), epochs=5000) # ??? f(stock_net) = ibb_net
    
    
    # test on test
    port_net_test = stock_net_test.iloc[port_index, 0:].T # we have S25 here
    ibb_learner_net_test = deep_learner.predict(port_net_test.as_matrix())# predict deep-learned ibb net change
    # calculate deep-learned ibb last price
    
    ibb_autoencoder = []
    price = 0
    for i in range(0,104):
        if i == 0:
            price = ibb_lp[104] # 2014.1.3 last price
        else:
            price = price + ibb_learner_net_test[i,0]
        ibb_autoencoder.append(price)

    diff = float(np.linalg.norm((ibb_autoencoder-ibb_lp_test.as_matrix())))
    error.append(diff) #25, 30, 35, ..., 65        

#%%    
    
mse =  [t /104 for t in error] 
plt.gca().invert_yaxis()
plt.plot(mse, list(range(5,56,5)))
plt.xlabel('Mean Square Error')
plt.ylabel('number of stocks in portfolio')
#%%

















