# Natural Language Toolkit: Encoders
#
# Copyright (C) 2001-2020 NLTK Project
# Author: Abilash Cheruvathur<abi.murali2009@gmail.com>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

"""
Natural Language Tool Kit Based Encoder

Given a series of text, the encoder encodes the text to form a dataframe of text vectors and has the capability to perform 
WordNet Lematizer, padding, remove stopwords, HTML tags, email addresses, numericals and custom regualr expressions. 

Example:

Input:
    
data['headline'][0]='This is a beautiful/ <day?'
data['headline'][1]='There are differences in time zones96'
0    This is a beautiful/ <day?
1    There are differences in time zones96
Output:
    
(   0  1  2  3  4
0  1  2  0  0  0
1  3  4  5  0  0, {'beautiful': 1, 'day': 2, 'difference': 3, 'time': 4, 'zone': 5})


@author: Abilash Cheruvathur <abi.murali2009@gmail.com>
"""
    	
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing import text
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer 
import pandas as pd
import re
