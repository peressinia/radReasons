#!/usr/bin/env Python3
"""Script to setup training, validation, and test splits for MIMIC-CXR TF-IDF vectors."""
#
# Version:  3.0c [19 Apr 24]
#
#   INPUT:  3 command line options
#               - string of reason file version
#               - integer index from 0-3 indicating reason file version
#               - integer > 1 indicating number of words in lexicon

#           1 csv file with the vectorized reasons 
#
#   OUTPUT: a csv file with MIMIC splits and CheXpert labels for the vectorized reasons
#           
#           
#   EXAMPLE:    $ getSplitsXY-TFIDF-v3_0c 3-2x 0 50
#
#              The above looks for a vectorized reason file named:
#                   'reason-vectors-TFIDF-50-NCF-v3-2x.csv'
#
#              It outputs the file:
#                   'xy-all-50-NCF-v3-2x.csv'
#
#               NOTE:   This script was originally used to break the reason vector
#                       file into SEPARATE files for use of the MIMIC CRX JPG 
#                       training, validation, test splits.  (Currently these
#                       lines are commented out as these splits did not work well
#                       for the text study here.)
#               

import sys
import numpy as np  
import pandas as pd
import argparse

# pd.options.mode.chained_assignment = None  # default='warn'

REASON_FILTER_VERSIONS = ['NCF', 'NCNF', 'CNF', 'CF']


##################################################################################
def getIndex (subject_arrays, study_array, subject, study):
    """Function to find index given subject and study numbers."""
##################################################################################    
    x = set(np.flatnonzero(subject_arrays == subject ))
    y = set(np.flatnonzero(study_array == study ))
    int_xy = [s for s in  x.intersection(y) ]
    if not int_xy:
        return(-1)
        #        raise ValueError('Error in GetIndex, pair not found subj={}, study={}.'.format(subject,study))
    else:
        return( min(int_xy) )
##################################################################################


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
##################################################################################
    
    reason_filter, reason_file_version, frequent_words_subset_size = doCL()
    TAG_STR = str(frequent_words_subset_size) + '-' + REASON_FILTER_VERSIONS[reason_filter] + '-v' + reason_file_version

    VEC_FILE_CSV_NAME = './reason-vectors-TFIDF-' + TAG_STR + '.csv'

#    LABEL_FILE_NAME_NEG = './mimic-cxr-2.0.0-negbio.csv'
    LABEL_FILE_NAME_CHEX = './mimic-cxr-2.0.0-chexpert.csv'
#    SPLIT_FILE_NAME = './mimic-cxr-2.0.0-split.csv'

 #   Y_TRAIN_NEG_FILENAME =  './y-train-neg-' + TAG_STR + '.csv'
 #   Y_VALI_NEG_FILENAME =   './y-vali-neg-' + TAG_STR + '.csv'
 #   Y_TEST_NEG_FILENAME =   './y-test-neg-' + TAG_STR + '.csv'
 
#    Y_TRAIN_CHEX_FILENAME = './y-train-chex-' + TAG_STR + '.csv'
#    Y_VALI_CHEX_FILENAME =  './y-vali-chex-' + TAG_STR + '.csv'
#    Y_TEST_CHEX_FILENAME =  './y-test-chex-' + TAG_STR + '.csv'

#    X_TRAIN_FILENAME =  './x-train-' + TAG_STR + '.csv'
#    X_VALI_FILENAME =   './x-vali-' + TAG_STR + '.csv'
#    X_TEST_FILENAME =   './x-test-' + TAG_STR + '.csv'

    ALL_DATA_FILENAME =     './xy-all-' + TAG_STR + '.csv'

    
    vectorData = pd.read_csv(VEC_FILE_CSV_NAME)
#    splitData = pd.read_csv(SPLIT_FILE_NAME)       


# NegBio Data has label appied to all    
# Load, fill blanks with zero, convert all to intergers
#    labelDataNeg = pd.read_csv(LABEL_FILE_NAME_NEG)
#    labelDataNeg = labelDataNeg.fillna(0)
#    labelDataNeg[list(labelDataNeg)] = labelDataNeg[list(labelDataNeg)].astype(int)
#    labelDataNeg['Findings'] = (labelDataNeg['No Finding'] != 1)*1

# ChesXpert Data has 2414 records with no labels - treat those as finding = 1    
# Load, fill blanks with zero, convert all to intergers
    labelDataChex = pd.read_csv(LABEL_FILE_NAME_CHEX)    
    labelDataChex = labelDataChex.fillna(0)    
    labelDataChex[list(labelDataChex)] = labelDataChex[list(labelDataChex)].astype(int)
#    labelDataChex['Findings'] = (labelDataChex['No Finding'] != 1)*1

# Add split column to vectorData        
#    vectorData['split'] ='none'

# Add NegBio findings column to vectorData    
#    vectorData['findingNeg'] ='0'

# Add CheXpert findings column to vectorData
    vectorData['findingChex'] ='0'

    for i in range(len(vectorData)):
#        indx = getIndex(splitData['subject_id'],splitData['study_id'], vectorData['subject_id'][i], vectorData['study_id'][i] )
#        if indx != -1:
#            vectorData.at[i,'split'] = splitData.at[indx,'split']
#        else:
#            print('Indx = -1 in splitdata loop  subj={}, study={}.'.format(vectorData['subject_id'][i],vectorData['study_id'][i]))
            
#        indx = getIndex(labelDataNeg['subject_id'],labelDataNeg['study_id'], vectorData['subject_id'][i], vectorData['study_id'][i] )
#        if indx != -1:
#            vectorData.at[i,'findingNeg'] = 0 if labelDataNeg.at[indx,'No Finding'] else 1
#        else:
#            print('Indx = -1 in labelDataNeg loop  subj={}, study={}.'.format(vectorData['subject_id'][i],vectorData['study_id'][i]))

        indx = getIndex(labelDataChex['subject_id'],labelDataChex['study_id'], vectorData['subject_id'][i], vectorData['study_id'][i] )
        if indx != -1:
            vectorData.at[i,'findingChex'] = 0 if labelDataChex.at[indx,'No Finding'] else 1
        else:
            print('Indx = -1 in findingChex loop  subj={}, study={}.'.format(vectorData['subject_id'][i],vectorData['study_id'][i]))


# Prepare Data Frames
    x_col_list= list(vectorData)
    x_col_list.remove('subject_id')
    x_col_list.remove('study_id')
    x_col_list.remove('search_term_code')
#    x_col_list.remove('split')
#    x_col_list.remove('findingNeg')
    x_col_list.remove('findingChex')
#    x_train = vectorData.loc[(vectorData['split']=='train'),x_col_list]
#    x_vali = vectorData.loc[(vectorData['split']=='validate'),x_col_list]
#    x_test = vectorData.loc[(vectorData['split']=='test'),x_col_list]

#    y_train_neg = vectorData.loc[(vectorData['split']=='train'),'findingNeg']
#    y_vali_neg = vectorData.loc[(vectorData['split']=='validate'),'findingNeg']
#    y_test_neg = vectorData.loc[(vectorData['split']=='test'),'findingNeg']
#    y_train_chex = vectorData.loc[(vectorData['split']=='train'),'findingChex']
#    y_vali_chex = vectorData.loc[(vectorData['split']=='validate'),'findingChex']
#    y_test_chex = vectorData.loc[(vectorData['split']=='test'),'findingChex']


# Write dataframes to disk    

#    y_train_neg.to_csv(Y_TRAIN_NEG_FILENAME, index=None)
#    y_vali_neg.to_csv(Y_VALI_NEG_FILENAME, index=None)
#    y_test_neg.to_csv(Y_TEST_NEG_FILENAME, index=None)


#    y_train_chex.to_csv(Y_TRAIN_CHEX_FILENAME, index=None)
#    y_vali_chex.to_csv(Y_VALI_CHEX_FILENAME, index=None)
#    y_test_chex.to_csv(Y_TEST_CHEX_FILENAME, index=None)

#    x_train.to_csv(X_TRAIN_FILENAME, index=None)
#    x_vali.to_csv(X_VALI_FILENAME, index=None)
#    x_test.to_csv(X_TEST_FILENAME, index=None)

    vectorData.to_csv(ALL_DATA_FILENAME,index=None)

    

if __name__ == '__main__':
    sys.exit(main())

