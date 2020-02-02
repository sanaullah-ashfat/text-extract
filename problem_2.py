# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:38:22 2020

@author: Sanaullah
"""
import pandas as pd
import numpy as np
import os 
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import hashing_trick
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence
from textblob.classifiers import NaiveBayesClassifier

df = pd.read_csv("G:/Tasks/Task2/roads.csv")

dff = df.fillna(method='bfill')


df1=pd.read_csv("G:/Tasks/Task2/user_raw_data.csv")
df2 =df1.iloc[:2206]
df2 =df2.fillna(method='bfill')

df3 = dff['Road']
df4= df2.join(dff,how='inner')

df4['new_clm'] = df4[['address', 'Road']].apply(tuple, axis=1)


train =df4['new_clm']

cl = NaiveBayesClassifier(train)


cl.classify("59/A Kalachandpur Gulshan-2,dhaka")



