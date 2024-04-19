#!/usr/bin/env Python3
"""Script to produce and associate TFIDF vectorized reasons for MIMIC-CXR reasons."""
# Version:  3.0c [19 Apr 24]
#
#   INPUT:  3 command line options
#               - string of reason file version
#               - integer index from 0-3 indicating reason file version
#               - integer > 1 indicating number of words in lexicon

#           1 csv file of top n terms with frequncy (n = 3rd command line option) 
#
#   OUTPUT: a csv file with the vectorized reasons with of top n terms with frequncy (n = 3rd command line option) 
#           
#           
#   EXAMPLE:    $ getReasonVectorsTFIDF-v2_1c 3-2x 0 50
#
#              The above looks for a lexicon file named:
#                   'BOW-words-top-50-NCF-v3-2x.csv'
#
#              Its output files are:
#                   'reason-vectors-TFIDF-50-NCF-v3-2x.csv'
#               
#
#
#


import sys
import csv
import nltk  
import pandas as pd
  
import re  
from sklearn.feature_extraction.text import TfidfTransformer
from collections import Counter
import argparse

REASON_FILTER_VERSIONS = ['NCF', 'NCNF', 'CNF', 'CF']


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
    
    REASON_FILE_NAME = './get-reasons/mimic-reasons-' + REASON_FILTER_VERSIONS[reason_filter] + '-v' + reason_file_version + '.csv'
    BOW_WORDS_FILE_NAME = './BOW-words-top-' + TAG_STR + '.csv'
    VEC_FILE_CSV_NAME = './reason-vectors-TFIDF-' + TAG_STR + '.csv'

    
    theSubjectID, theStudyID, theSearchID, theReasons = ([] for i in range(4))
    with open(REASON_FILE_NAME, newline='') as theReasonFile:
         reasonReader = csv.reader(theReasonFile)
         reasonFileHeader = next(reasonReader)          # eat the header line
         for row in reasonReader:
             theReasons.append(row[-1])
             theStudyID.append(row[1])
             theSubjectID.append(row[0])
             theSearchID.append(row[2])
    
    for i in range(len(theReasons )):
        theReasons [i] = theReasons [i].lower()
        theReasons [i] = re.sub(r'\W',' ',theReasons [i])
        theReasons [i] = re.sub(r'\s+',' ',theReasons [i])

    theWordList = []
    with open(BOW_WORDS_FILE_NAME, newline='') as theWordFile:
         wordReader = csv.reader(theWordFile)
         wordFileHeader = next(wordReader)          # eat the header line
         for row in wordReader:
             theWordList.append(row[0])
        
# create a vector of theWordList Frequencies 
    reasonVectors = []
    for reason in theReasons:
        reason_tokens = nltk.word_tokenize(reason)
        rVec = []
        xCounter = Counter(reason_tokens)
        for word in theWordList:
            if word in xCounter:
                rVec.append(xCounter[word])
            else:
                rVec.append(0)
        reasonVectors.append(rVec)

    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(reasonVectors)
# Test print out
#    df_idf = pd.DataFrame(tfidf_transformer.idf_, index=theWordList,columns=["idf_weights"])
#    print(df_idf)
    tf_idf_vector=tfidf_transformer.transform(reasonVectors)

    #get tfidf vector for first document and print scores
#    first_document_vector=tf_idf_vector[6] 
#    df = pd.DataFrame(first_document_vector.T.todense(), index=theWordList, columns=['tf-idf'])
#    df.sort_values(by=['tf-idf'],ascending=False)
#    print(df)


    csvHeader = reasonFileHeader[0:3]
    csvHeader.extend(['w_{}'.format(s) for s in theWordList])
    with open(VEC_FILE_CSV_NAME, 'w') as theOutFile:
        fileWriter = csv.writer(theOutFile)
        fileWriter.writerow(csvHeader)
        for i in range(len(reasonVectors)):
            df = pd.DataFrame(tf_idf_vector[i].T.todense(), columns=['tf-idf'])
            aVec = df['tf-idf'].tolist()
            fileWriter.writerow([theSubjectID[i], theStudyID[i], theSearchID[i]]+ aVec)
               


if __name__ == '__main__':
    sys.exit(main())

