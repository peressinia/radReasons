#!/usr/bin/env Python3
"""Script to run Naive Bayes on MIMIC-CXR BOW TF-IDF vectors."""
# Script:   doNaiveBayesTFIDF
# Version:  1.0 [20 Mar 23]
#

import sys
import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
import seaborn as sns

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

Y_TRAIN_NEG_FILENAME =  './y-train-neg.csv'
Y_VALI_NEG_FILENAME =   './y-vali-neg.csv'
Y_TEST_NEG_FILENAME =   './y-test-neg.csv'

Y_TRAIN_CHEX_FILENAME = './y-train-chex.csv'
Y_VALI_CHEX_FILENAME =  './y-vali-chex.csv'
Y_TEST_CHEX_FILENAME =  './y-test-chex.csv'

X_TRAIN_FILENAME =      './x-train.csv'
X_VALI_FILENAME =       './x-vali.csv'
X_TEST_FILENAME =       './x-test.csv'


ALL_DATA_FILENAME =     './xy-all.csv'

# TEST_PERCENT = 0.20

##################################################################################



##################################################################################
def main():
    """Main entry point for the script."""
 
# Importing the dataset
#    X = pd.read_csv(X_TRAIN_FILENAME)
#    y = pd.read_csv(Y_TRAIN_NEG_FILENAME)
    X_train = pd.read_csv(X_TRAIN_FILENAME)
    X_test = pd.read_csv(X_TEST_FILENAME)
    y_train = pd.read_csv(Y_TRAIN_NEG_FILENAME)
    y_test = pd.read_csv(Y_TEST_NEG_FILENAME)
    classifier_list = [MultinomialNB(), BernoulliNB(), GaussianNB()] 
    print('Naive Bayes TF-IDF Run: X_train shape = {}.'.format(X_train.shape))
    print('\t Training Num = {}, Test Num = {}.'.format(len(X_train) ,len(X_test)))
    
    
    for classifier in classifier_list:
        print('===========================================================')
#        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_PERCENT, random_state = None)
        classifier.fit(X_train, y_train.values.ravel())
        y_pred = classifier.predict(X_test)
        ac = accuracy_score(y_test,y_pred)
        cm = confusion_matrix(y_test, y_pred)
        print('{0} Model accuracy score: {1:0.4f}'. format(classifier,ac))
        cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], index=['Predict Positive:1', 'Predict Negative:0'])                                 
        plt1 = sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
        theFig = plt1.get_figure()
        theFig.savefig('ConfusionMat-{}.png'.format(classifier.__class__.__name__)) 
        theFig.clf() # this clears the figure
        y_pred_train = classifier.predict(X_train)
        print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
        print(classification_report(y_test, y_pred))
        TP = cm[0,0]
        TN = cm[1,1]
        FP = cm[0,1]
        FN = cm[1,0]
        # print classification accuracy
        classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
        print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
        # print classification error
        classification_error = (FP + FN) / float(TP + TN + FP + FN)
        print('Classification error : {0:0.4f}'.format(classification_error))
        # print precision score
        precision = TP / float(TP + FP)
        print('Precision : {0:0.4f}'.format(precision))
        # print recall score
        recall = TP / float(TP + FN)
        print('Recall or Sensitivity : {0:0.4f}'.format(recall))
        # print true positive
        true_positive_rate = TP / float(TP + FN)
        print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
        # print false postive
        false_positive_rate = FP / float(FP + TN)
        print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
        # print Specificity
        specificity = TN / (TN + FP)
        print('Specificity : {0:0.4f}'.format(specificity))
    print('===========================================================')


if __name__ == '__main__':
    sys.exit(main())



