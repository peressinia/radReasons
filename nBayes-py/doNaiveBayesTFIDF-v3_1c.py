#!/usr/bin/env Python3
"""Script to run Naive Bayes (several versions) on MIMIC-CXR TF-IDF vectors."""
# Script:   doNaiveBayesTFIDF
# Version:  1.0 [20 Mar 23] uses MIMIC training set for training & MIMIC test set for testing.
# Version:  2.0 [20 Jan 24] REASON_FILTER_VERSIONS added for getReason-v3-2:  ['NCF', 'NCNF', 'CNF']
#               uses MIMIC training set for training & MIMIC test set for testing.
# Version:  2.1 [29 Jan 24] adds lift plots, etc. doNaiveBayes-MultNom-withLift-v2_1.py
#               [30 Jan 24] uses argparse
# Version:  2.1x[30 Jan 24] uses all MIMIC data; randomly assigns training/test sets; MIMIC splits NOT used
# Version:  2.2x[05 Feb 24] added printouts of high prob vector
# Version:  2.3x [16 Feb 24] added OUT_STAT_FILENAME = './NB-Stat-Tally.csv' to output stats
# Version:  3.0 [22 Mar 24] based on v2.3;  added final stats (ROC_AUC) to coodinate with do_BERT
#               added 'CF' option and using only random splits. 
# Version:  3.1 [02 Apr 24] based on v3.0: fixed TP <-> TN error  
#
#


import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
import myKDS as mK
import time
from time import strftime
import argparse, os.path

REASON_FILTER_VERSIONS = ['NCF', 'NCNF', 'CNF', 'CF']
TEST_PERCENT = 0.10
RANDOM_STATE = 42


##################################################################################
def doCL():
    """Parse the command line."""
##################################################################################
    parser = argparse.ArgumentParser(description='Word-tokenize MIMIC-CXR radiology report INDICATIONS with frequencies.')
    parser.add_argument('reason_file_version', 
                    help='string representing version of reason file',
                    )
    parser.add_argument('reason_filter_index', help='integer from 0 to ' + str(len(REASON_FILTER_VERSIONS)) + ' indicating a filter type:' + ''.join(str(REASON_FILTER_VERSIONS))  ,
                    type=int)
    parser.add_argument('subset_size', help='integer indicating the number of words to retain',
                    type=int)

    args = parser.parse_args()
    return args.reason_filter_index, args.reason_file_version, args.subset_size  
##################################################################################


##################################################################################
def main():
    """Main entry point for the script."""
    
    reason_filter, reason_file_version, frequent_words_subset_size = doCL()
    TAG_STR = str(frequent_words_subset_size) + '-' + REASON_FILTER_VERSIONS[reason_filter] + '-v' + reason_file_version

#    Y_TRAIN_CHEX_FILENAME = './y-train-chex-' + TAG_STR + '.csv'
#    Y_VALI_CHEX_FILENAME =  './y-vali-chex-' + TAG_STR + '.csv'
#    Y_TEST_CHEX_FILENAME =  './y-test-chex-' + TAG_STR + '.csv'

#    X_TRAIN_FILENAME =  './x-train-' + TAG_STR + '.csv'
#    X_VALI_FILENAME =   './x-vali-' + TAG_STR + '.csv'
#    X_TEST_FILENAME =   './x-test-' + TAG_STR + '.csv'

    ALL_DATA_FILENAME =     './xy-all-' + TAG_STR + '.csv'
    OUT_STAT_FILENAME = './NB-Stat-Tally.csv'

    
# Importing the dataset
#    X_train = pd.read_csv(X_TRAIN_FILENAME)
#    X_test = pd.read_csv(X_TEST_FILENAME)
#    y_train = pd.read_csv(Y_TRAIN_CHEX_FILENAME)
#    y_test = pd.read_csv(Y_TEST_CHEX_FILENAME)

    allData = pd.read_csv(ALL_DATA_FILENAME)
    X = allData.iloc[:,3:-3]
    y = allData.iloc[:,-1:] #-1 is Chexpert col 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_PERCENT, random_state = RANDOM_STATE)


    classifier_list = [MultinomialNB(), BernoulliNB(), GaussianNB()] 
    print('Naive Bayes TF-IDF Run (CheXpert; XSplits): X_train shape = {}.'.format(X_train.shape))
    print('\t Version Tag: {}.'.format(TAG_STR))
    print('\t Training Num = {}, Test Num = {}.'.format(len(X_train) ,len(X_test)))
    print('\t Test Percent: {}.'.format(TEST_PERCENT))
    dataSaveDF = pd.DataFrame(columns = ['verTag', 'model', 'reason-filter', 'reason-version',
                                         'Size', 'Splits', 'nTrain', 'nTest', 
                                         'accTest', 'accTrain', 'MCC', 'RocAuc',
                                         'f1-0', 'f1-1',
                 'num-0', 'num-1', 'precision', 'recall', 'tp-rate', 'fp-rate', 'specificity', 'decile1-n', 'd1-pmin', 'd1-pmax', 
                 'lift1', 'ks-stat', 'ks-decile', 'TP', 'TN', 'FP', 'FN', 'timestamp'])
    
    for classifier in classifier_list:
        start_time = time.time()
        print('===========================================================')
        output_header = classifier.__class__.__name__ + '-TFIDF-xSplits-' + TAG_STR   #'-n' + str(X.shape[1])
        dataSave = []

        dataSave = [output_header, classifier.__class__.__name__, REASON_FILTER_VERSIONS[reason_filter], reason_file_version, 
                    frequent_words_subset_size,'Random (test='+str(TEST_PERCENT)+')',len(X_train),len(X_test)]


        classifier.fit(X_train, y_train.values.ravel())
        y_pred = classifier.predict(X_test)
        ac = accuracy_score(y_test,y_pred)
        cm = confusion_matrix(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        print('{0} Model accuracy score: {1:0.4f}'. format(classifier,ac))
        print('Total MCC: %.3f' % mcc)
        cm_matrix = pd.DataFrame(data=cm, columns=['Actual Negative:0', 'Actual Positive:1'], index=['Predict Negative:0', 'Predict Positive:1'])                                 
        plt1 = sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
        theFig = plt1.get_figure()
        theFig.savefig('ConfusionMat-{}.png'.format(output_header)) 
        theFig.clf() # this clears the figure
        y_pred_train = classifier.predict(X_train)
        dataSave.extend([accuracy_score(y_train, y_pred_train),ac,mcc])
        print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

        RocAuc = roc_auc_score(y_test,y_pred)
        print('ROC_AUC Score: {0:0.4f}'.format(RocAuc))
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        plt.figure(figsize=(7,7))
        output_file = './ROC-Plot-' + output_header
        Roc_ti_String = classifier.__class__.__name__ + ' with ' + REASON_FILTER_VERSIONS[reason_filter] + ' (Num Terms = ' + str(frequent_words_subset_size) +')'
        from plot_metric.functions import BinaryClassification # Visualisation with plot_metric
        bc = BinaryClassification(y_test, y_pred, labels=["Class 1", "Class 2"])
        bc.plot_roc_curve(plot_threshold=False,title='ROC Curve: ' + Roc_ti_String)
        plt.savefig(output_file)
        #plt.show()
        plt.clf()
        
        cRep = classification_report(y_test, y_pred,output_dict=True)
        dataSave.extend([RocAuc,cRep['0']['f1-score'],cRep['1']['f1-score'],cRep['0']['support'],cRep['1']['support']])
        print(classification_report(y_test, y_pred))
        
        TN, FP, FN, TP = cm.ravel()
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
        dataSave.extend([precision,recall,true_positive_rate,false_positive_rate,specificity])
    
    
        y_prob = classifier.predict_proba(X_test)
        y_truth = y_test.iloc[:,0].values
        # next line supresses SettingwithCopyWarning in the line after
        with pd.option_context('mode.chained_assignment', None):
            X_test['predicted_prob'] = y_prob[:,1]

        # CUMMULATIVE GAIN PLOT
        theFig = mK.plot_cumulative_gain(y_truth , y_prob[:,1])  
        theFig.savefig('fig-CumGain-{}.pdf'.format(output_header)) 
        theFig.clf() # this clears the figure
        
        # LIFT PLOT
        theFig = mK.plot_lift(y_truth , y_prob[:,1])  
        theFig.savefig('fig-lift-{}.pdf'.format(output_header)) 
        theFig.clf() # this clears the figure
        
        # KS Stat PLOT
        theFig = mK.plot_ks_statistic(y_truth , y_prob[:,1])  
        theFig.savefig('fig-ksStat-{}.pdf'.format(output_header)) 
        theFig.clf() # this clears the figure
        
        dt, theFig = mK.report(y_truth, y_prob[:,1],plot_style='ggplot',labels=False)
        theFig.savefig('fig-4lift-{}.pdf'.format(output_header)) 
        theFig.clf() # this clears the figure
        
        
        xx=mK.decile_table(y_truth, y_prob[:,1],labels=False)    
        ksmx = xx.KS.max()
        ksdcl = xx[xx.KS == ksmx].decile.values
        print('===========================================================')
        decToPrint = 0
        print('Decile {}: count = {}, prob min = {}, prob max = {}.'.format(xx.iloc[decToPrint,0],
                                                                            xx.iloc[decToPrint,5],
                                                                            xx.iloc[decToPrint,1],
                                                                            xx.iloc[decToPrint,2]))
        print('lift @ decile {} = {}.'.format(decToPrint+1,xx['lift'].iloc[0]))
        print('KS Stat: {} @ decile {}.'.format(str(ksmx),str(list(ksdcl)[0])))
        print('===========================================================\n')
        dataSave.extend([xx.iloc[decToPrint,5],xx.iloc[decToPrint,1],xx.iloc[decToPrint,2],xx['lift'].iloc[0],ksmx,ksdcl[0]])
        dataSave.extend([TP,TN,FP,FN])

        print('===========================================================')
        df = X_test.sort_values(by='predicted_prob', ascending=False)
        df = df.drop(['predicted_prob'], axis=1)
        X_test = X_test.drop(['predicted_prob'], axis=1) # for the loop not to error next time classifier invoked
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
        print('===========================================================\n')
        


        print('===========================================================')        
        minutes, seconds = divmod((time.time() - start_time), 60)
        print('\nRun time:    --- {} minutes and {} seconds ---'.format(int(minutes), int(seconds)))
        print('===========================================================')

        dataSave.extend([strftime("%d %b %Y %H:%M:%S", time.localtime(time.time()))])
        dataSaveDF.loc[-1] = dataSave            # add row
        dataSaveDF.index = dataSaveDF.index + 1  # shifting index
        dataSaveDF = dataSaveDF.sort_index()     # sorting by index

    print('===========================================================')
    if(os.path.isfile(OUT_STAT_FILENAME)):
        dataSaveDF.to_csv(OUT_STAT_FILENAME, encoding='utf-8', mode='a', index=False, header=False)
    else:   
        dataSaveDF.to_csv(OUT_STAT_FILENAME, encoding='utf-8', index=False, header=True)

if __name__ == '__main__':
    sys.exit(main())



