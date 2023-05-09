#!/usr/bin/env Python3
"""Script to produce and associate BOW vectors with MIMIC-CXR reason-for-exam."""
# Script:   getReasonVectorsTFIDF.py
# Version:  1.0 [20 Mar 23]
#

import sys
import csv
import nltk  
#nltk.download('punkt')
#nltk.download('stopwords')
#import numpy as np
import pandas as pd
  
import re  
#import heapq
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from collections import Counter
 

#REASON_FILE_NAME = './mimic-reasons-BERT-v5.csv'
REASON_FILE_NAME = './mimic-reasons-BERT-local.csv'
BOW_WORDS_FILE_NAME = './BOW-words.csv'
VEC_FILE_CSV_NAME = './BOW-reason-TFIDFvectors.csv'



##################################################################################
##################################################################################
def main():
    """Main entry point for the script."""
    

    
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

