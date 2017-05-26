#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 16:12:35 2017

@author: phil
"""

#%%

import numpy as np
import pandas as pd

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt


## load data
print("Loading the data ...")
full_data = pd.read_csv('full_data/net_change.csv', header=0).T # inlcude date

data = full_data.drop(["Unnamed: 0"]).dropna(axis=0, how='any') # axis=0: x-axis

train_data = data.iloc[0:, 0:104] # 2012.01 ~ 2013.12
test_data = data.iloc[0:, 104:226] # 2014.01 ~ 2016.04


'''
for train_data, test_data

    row: stock
    column: data
'''
#%%

