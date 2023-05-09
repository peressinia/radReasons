#!/usr/bin/env Python3
"""Script to get lift for Naive Bayes (multinomial) on MIMIC-CXR BOW TF-IDF vectors."""
# Script:   doNaiveBayes-MultNom-withLift
# Version:  1.0
#

import sys
#import matplotlib.pyplot as plt
#import numpy as np  
# import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
#from sklearn.metrics import classification_report
#import seaborn as sns
import kds
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
#from sklearn.naive_bayes import GaussianNB
#from sklearn.model_selection import GridSearchCV

Y_TRAIN_NEG_FILENAME =  './y-train-neg.csv'
Y_VALI_NEG_FILENAME =   './y-vali-neg.csv'
Y_TRAIN_CHEX_FILENAME = './y-train-chex.csv'
Y_VALI_CHEX_FILENAME =  './y-vali-chex.csv'
X_TRAIN_FILENAME =      './x-train.csv'
X_VALI_FILENAME =       './x-vali.csv'
ALL_DATA_FILENAME =     './xy-all.csv'

TEST_PERCENT = 0.20

##################################################################################



##################################################################################
def main():
    """Main entry point for the script."""
 
# Importing the dataset
    X = pd.read_csv(X_TRAIN_FILENAME)
    y = pd.read_csv(Y_TRAIN_NEG_FILENAME)
#    classifier_list = [MultinomialNB(), BernoulliNB(), GaussianNB()]   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_PERCENT, random_state = 41)
#    classifier = MultinomialNB(class_prior=[0.1, 0.9])
#    classifier = BernoulliNB()
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train.values.ravel())
    y_pred = classifier.predict(X_test)
    ac = accuracy_score(y_test,y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], index=['Predict Positive:1', 'Predict Negative:0'])                                 
    print('{0} Model accuracy score: {1:0.4f}'. format(classifier,ac))
    print(cm_matrix)
    print(classifier.class_prior)    
    y_prob = classifier.predict_proba(X_test)
    X_test['predicted_prob'] = y_prob[:,1]

# CUMMULATIVE GAIN PLOT
    kds.metrics.plot_cumulative_gain(y_test.loc[:,'findingNeg'].to_numpy(), y_prob[:,1])    
#    theFig = plt1.get_figure()
#    theFig.savefig('Cum-Gain-{}-n{}.png'.format(classifier.__class__.__name__),X.shape[1]) 
#    theFig.clf() # this clears the figure
    kds.metrics.plot_lift(y_test.loc[:,'findingNeg'].to_numpy(), y_prob[:,1])    
    kds.metrics.plot_ks_statistic(y_test.loc[:,'findingNeg'].to_numpy(), y_prob[:,1])    
    kds.metrics.decile_table(y_test.loc[:,'findingNeg'].to_numpy(), y_prob[:,1])    
    xx=kds.metrics.report(y_test.loc[:,'findingNeg'].to_numpy(), y_prob[:,1],plot_style='ggplot')
    print('===========================================================')
    decToPrint = 0
    print('Decile {}: count = {}, prob min = {}, prob max = {}.'.format(xx.iloc[decToPrint,0],
                                                                        xx.iloc[decToPrint,5],
                                                                        xx.iloc[decToPrint,1],
                                                                        xx.iloc[decToPrint,2]))
    print('===========================================================')

    print('===========================================================')
    df = X_test.sort_values(by='predicted_prob', ascending=False)
    df = df.drop(['predicted_prob'], axis=1)
    df = df.head(100)
    df_dict = dict(
        list(
            df.groupby(df.index)
        )
    )
    for k, v in df_dict.items():               # k: name of index, v: is a df
        check = v.columns[(v != 0).any()]
        if len(check) > 0:
            print((k, check.to_list()))
print('===========================================================')        


if __name__ == '__main__':
    sys.exit(main())


