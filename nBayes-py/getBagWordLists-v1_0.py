#!/usr/bin/env Python3
"""Script to word-tokenize MIMIC-CXR radiology report REASON FOR EXAM with frequencies."""
# Script:   getBagWordLists.py
# Version:  1.0 [20 Mar 23]

import sys
import csv
import nltk  
#nltk.download('punkt')
#nltk.download('stopwords')
import numpy as np  
import re  
import heapq

# WORD_FILE_NAME = './mimic-reasons-BERT-v5.csv'
WORD_FILE_NAME = './mimic-reasons-BERT-local.csv'
ALL_OUT_FILE_NAME = './BOW-words-all.csv'
FREQUENT_WORDS_SUBSET_SIZE = 5
SUB_OUT_FILE_NAME = './BOW-words-top-' +str(FREQUENT_WORDS_SUBSET_SIZE) + '.csv'
ALL_PUNCTUATION_SET = {'!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', \
                   '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', \
                       '_', '`', '{', '|', '}', '~'}
KEPT_PUNCTUATION_SET = {'|', '_'}
REMOVED_PUNCTUATION_SET =  ALL_PUNCTUATION_SET - KEPT_PUNCTUATION_SET
OUR_STOP_WORD_SET = {'pls', 'a',
 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be',
 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't",
 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn',
 "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself',
 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn',
 "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off',
 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'please', 'pls', 're', 's', 'same', 'shan',
 "shan't", 'she', "she's", 'should', "should've", 'shouldn', "shouldn't", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the',
 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until',
 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 'what', 'when', 'where', 'which', 'while', 'who',
 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", "you're", "you've", 'your',
 'yours', 'yourself', 'yourselves',
 "ct", "ago", "left", "right", "hx", "l", "r", "patient", "x", "history", "sided", "side", "previous", "per", "possible", 
 "would", "also", "pt", "needs", "today", "finding", "feeling", "day", "mri", "dr", "stairs", "like", "known", "newly",
 "presents", "'s'", "vs", "night", "done", "time", "due", "past", "years", "days", "showed", "work", "..", "unfortunately",
 "going", "requested", "still", "yesterday", "reports", "door", "background", "bicycle", "thanks", "ty", "thank", "glass"
 "old", "year", "chest", "eval", "evaluate","assess","evaluation","status", "w", "c", "crx", "presenting", "h","b","l","prior"
 "1","2","3", "f","week", "weeks"}



##################################################################################
##################################################################################
def main():
    """Main entry point for the script."""
    
    csvHeader = ['term','freq']        
    
    theReasons = []
    with open(WORD_FILE_NAME, newline='') as theWordFile:
         wordReader = csv.reader(theWordFile)
         wordFileHeader = next(wordReader)          # eat the header line
         for row in wordReader:
             theReasons.append(row[-1])
    
    for i in range(len(theReasons )):
        theReasons [i] = theReasons [i].lower()
        theReasons [i] = re.sub(r'\W',' ',theReasons [i])
        theReasons [i] = re.sub(r'\s+',' ',theReasons [i])
        
    stop_words = set(nltk.corpus.stopwords.words('english')).union(OUR_STOP_WORD_SET)

    termFreqSet = {}
    for reason in theReasons:
        tokens = nltk.word_tokenize(reason)
        for token in tokens:
            if token in stop_words:
                pass                    
            elif token not in termFreqSet.keys():
                termFreqSet[token] = 1
            else:
                termFreqSet[token] += 1    

    with open(ALL_OUT_FILE_NAME, 'w') as theOutFile:
        fileWriter = csv.writer(theOutFile)
        fileWriter.writerow(csvHeader)
        for key, value in termFreqSet.items():
            fileWriter.writerow([key, value])
               
    termFreqSubSet = heapq.nlargest(FREQUENT_WORDS_SUBSET_SIZE, termFreqSet, key=termFreqSet.get)
    
    with open(SUB_OUT_FILE_NAME, 'w') as theOutFile:
        fileWriter = csv.writer(theOutFile)
        fileWriter.writerow(csvHeader)
        for word in termFreqSubSet:
            fileWriter.writerow([word, termFreqSet[word]])

    reasonVectors = []
    for reason in theReasons:
        reason_tokens = nltk.word_tokenize(reason)
        rVec = []
        for term in termFreqSubSet:
            if term in reason_tokens:
                rVec.append(1)
            else:
                rVec.append(0)
        reasonVectors.append(rVec)

    reasonVectors = np.asarray(reasonVectors)



if __name__ == '__main__':
    sys.exit(main())

