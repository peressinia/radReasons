#!/usr/bin/env Python3
"""Script to get stast from mimic-reasons-BERT-v6.csv file."""
#
# Script:   doReasonsFileStats.py
#               version: 1.0 
#
# 
#
#
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATAFILE_NAME = 'mimic-reasons-BERT-v6.csv'





#######################################################################
# Get data
#######################################################################
theData = pd.read_csv(DATAFILE_NAME)
# Report the number of sentences.
print('Total number of reasons: {:,}\n'.format(theData.shape[0]))


df_neg = theData.groupby(['split'])[['neg_label']].sum()
df_neg = df_neg.rename(columns={'neg_label': 'finding'})
df_chx = theData.groupby(['split'])[['chex_label']].sum()
df_chx =  df_chx.rename(columns={'chex_label': 'finding'})

df_neg['no_finding'] = 0
df_chx['no_finding'] = 0

# df_ttl_neg = theData.groupby(['split'])[['neg_label']].count()
# df_ttl_chx = theData.groupby(['split'])[['chex_label']].count()

df_neg['total'] = theData.groupby(['split'])[['neg_label']].count().iloc[:,0]
df_chx['total'] = theData.groupby(['split'])[['chex_label']].count().iloc[:,0]


df_neg['no_finding'] = df_neg['total'] - df_neg['finding']
df_chx['no_finding'] = df_chx['total'] - df_chx['finding']

df_neg['finding_prct'] = df_neg['finding'] / df_neg['total']
df_chx['finding_prct'] = df_chx['finding'] / df_chx['total']

print(df_chx)
print(df_neg)



# get length of all the text in the dataframe
#seq_len_premise = [len(i.split()) for i in df['text']]
#pd.Series(seq_len_premise).hist(bins = 25)
df = theData.rename(columns={'chex_label': 'Finding'})
percentage = [32, 68]
sns.countplot(x=df['Finding'])
