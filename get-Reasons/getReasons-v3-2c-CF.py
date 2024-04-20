#!/usr/bin/env Python3
"""Script to extract MIMIC-CXR radiology report REASON FOR EXAM field with CF filter."""
#
# Version:  3.2c [19 Apr 24]
#
#   INPUT:  5 files with the results of the GREP search 
#                   'grep-indication.txt', 
#                   'grep-reasonForExam.txt', 
#                   'grep-reasonForExamination.txt', 
#                   'grep-history.txt', 
#                   'grep-clinicalInfo.txt'
#
#           3 files from the MIMIC set
#                   'mimic-cxr-2.0.0-negbio.csv'
#                   'mimic-cxr-2.0.0-chexpert.csv'
#                   'mimic-cxr-2.0.0-split.csv'
#
#   OUTPUT: a cased, filtered Indications csv file with column headers as follows:
#           [subject_id, study_id, search_term_code, split, neg_label, chex_label, reason_for_exam]
#           
#           
#   EXAMPLE:    $ getReasons-v3-2-CF
#
#              The above looks for the 8 files list above ...
#
#              It creates the file of reasons:
#                   'mimic-reasons-CF-v3-2x.csv'
# 

import sys
import time
import os
import csv
import re
import numpy as np
import pandas as pd

REASON_SEARCH_STRINGS = ['INDICATION:', 'REASON FOR EXAM:', 'REASON FOR EXAMINATION:', 'HISTORY:', 'CLINICAL INFORMATION:']
REASON_SEARCH_TERM_CODES = [0, 1, 2, 3, 4]
DIRECTORY_FILE_NAMES = ['./grep-indication.txt', './grep-reasonForExam.txt', './grep-reasonForExamination.txt', './grep-history.txt', './grep-clinicalInfo.txt']
OUT_FILE_NAME = 'mimic-reasons-CF-v3-2.csv'

LABEL_FILE_NAME_NEG = './mimic-cxr-2.0.0-negbio.csv'
LABEL_FILE_NAME_CHEX = './mimic-cxr-2.0.0-chexpert.csv'
SPLIT_FILE_NAME = './mimic-cxr-2.0.0-split.csv'


#############################################################
def filter_reason(x):
    """Function to filter and replace abbreviation in REASON FOR EXAM string."""

#    x = x.casefold()    # commented out only for CF version [since filered it comes though this function]
#   and function below is used instead of x = x.replace
#
    def x_replace(x, old, new):
        # Replacing all occurrences of substring s1 with s2
        x = re.sub(r'(?i)'+old, new, x)
        return x

# Next Set replace abbviations for sex and standardize
    x = x_replace(x, '___f ', ' female ')
    x = x_replace(x, '___/f ', ' female ')
    x = x_replace(x, '___ f ', ' female ')
    x = x_replace(x, '___ yo f ', ' female ')
    x = x_replace(x, '___ yr f ', ' female ')
    x = x_replace(x, '___ y f ', ' female ')
    x = x_replace(x, '___ y/o f ', ' female ')
    x = x_replace(x, '___ year f ', ' female ')
    x = x_replace(x, '___-year old f ', ' female ')
    x = x_replace(x, '___-year-old f ', ' female ')
    x = x_replace(x, '___ year old f ', ' female ')
    x = x_replace(x, '___ year-old f ', ' female ')
    x = x_replace(x, '___ years old f ', ' female ')
    x = x_replace(x, '___ y.o f ', ' female ')
    x = x_replace(x, '___ y.o. f ', ' female ')
    x = x_replace(x, ' ___ f ', ' female ')
    x = x_replace(x, 'y/o f ', ' female ')
    x = x_replace(x, 'woman', ' female ')

    x = x_replace(x, '___m ', ' male ')
    x = x_replace(x, '___/m ', ' male ')
    x = x_replace(x, '___ m ', ' male ')
    x = x_replace(x, '___ yo m ', ' male ')
    x = x_replace(x, '___ yr m ', ' male ')
    x = x_replace(x, '___ y m ', ' male ')
    x = x_replace(x, '___ y/o m ', ' male ')
    x = x_replace(x, '___ year m ', ' male ')
    x = x_replace(x, '___-year old m ', ' male ')
    x = x_replace(x, '___-year-old m ', ' male ')
    x = x_replace(x, '___ year old m ', ' male ')
    x = x_replace(x, '___ year-old m ', ' male ')
    x = x_replace(x, '___ years old m ', ' male ')

    x = x_replace(x, '___ y.o m ', ' male ')
    x = x_replace(x, '___ y.o. m ', ' male ')
    x = x_replace(x, ' ___ m ', ' male ')
    x = x_replace(x, 'y/o m ', ' male ')
    x = x_replace(x, 'man', ' male ')
    
    x = x_replace(x, 'yo ', ' ')               # ver 2.0
    x = x_replace(x, 'y o ', ' ')              # ver 2.0


# Next Set is for other known abbriviations
    x = x_replace(x, 's/p', ' status post ')  #ver 2.0
    x = x_replace(x, ' sp ', ' status post ')  #ver 2.0
    x = x_replace(x, 's p ', ' status post ')  #ver 2.0

    x = x_replace(x, 'c/o', ' complains of ')  #ver 1.1 pullReason
    x = x_replace(x, ' w/', ' with ')  #ver 1.1 pullReason

    x = x_replace(x, 'r/o', ' rule_out ')
    x = x_replace(x, ' ro ', ' rule_out ')
    x = x_replace(x, 'rule out', ' rule_out ')

    x = x_replace(x, 'f/u', ' follow_up ')
    x = x_replace(x, 'follow up', ' follow_up ')

    x = x_replace(x, 'ptx', ' pneumothorax ')

    x = x_replace(x, 'oxygen requirement', ' oxygen_requirement ')
    x = x_replace(x, 'o2 requirement', ' oxygen_requirement ')
    x = x_replace(x, 'o2', 'oxygen')
    
    x = x_replace(x, 'questionable', 'question')
    x = x_replace(x, 'confirming', 'confirm')

    x = x_replace(x, 'chf', ' cong_heart_failure ')
    x = x_replace(x, 'congestive heart failure', ' cong_heart_failure ')

    x = x_replace(x, 'sob', ' shortness_of_breath ')
    x = x_replace(x, 'shortness of breath', ' shortness_of_breath ')
    x = x_replace(x, 'pulmonary', ' pulm ')

    x = x_replace(x, 'endotracheal tube', ' endotracheal_tube ')
    x = x_replace(x, 'et tube', ' endotracheal_tube ')
    x = x_replace(x, 'ett tube', ' endotracheal_tube ')

    x = x_replace(x, ' ca ', ' cancer ')
    x = x_replace(x, ' ca.', ' cancer ')    

#    x = x_replace(x, 'coronary artery disease', ' cad ')
    x = x_replace(x, ' cad ', 'coronary artery disease')

    x = x_replace(x, 'fxs', ' fracture ')
    x = x_replace(x, 'fx', ' fracture ')
    x = x_replace(x, 'fractures', ' fracture ')

#    x = x_replace(x, 'end-stage renal disease', ' esrd ') 
#    x = x_replace(x, 'end stage renal disease', ' esrd ')     
    x = x_replace(x, ' esrd ', ' end stage renal disease ') 

    x = x_replace(x, 'fluid overload', ' fluid_overload ')
    x = x_replace(x, 'volume overload', ' volume_overload ')    

    x = x_replace(x, 'htn', ' hypertension ')

    x = x_replace(x, 'pulmonary embolism', ' pe ')

    x = x_replace(x, 'pna', ' pneumonia ')  # space before "pna" removed v. 1.2

    x = x_replace(x, 'nasogastric tube', ' nasogastric_tube ')
    x = x_replace(x, 'ngt', ' nasogastric_tube ')
    x = x_replace(x, 'ng tube', ' nasogastric_tube ')

#    x = x_replace(x, 'right lower lobe', ' rll ')        
    x = x_replace(x, 'rll',' right lower lobe ')        

    x = x_replace(x, 'avr', ' aortic valve_replacement ')        
    x = x_replace(x, 'mvr', ' mitral valve_replacement ')        
    x = x_replace(x, 'tvr', ' tricuspid valve_replacement ')        
    x = x_replace(x, 'valve replacement', ' valve_replacement ')        


    x = x_replace(x, '___', ' ')
    x = x_replace(x, 'year', ' ')    # ver 2.0
    x = x_replace(x, 'old', ' ')     # ver 2.0
   
  #  x = re.sub(r'[^a-zA-Z0-9]', ' ', x)  # ver 2.0; removed ver 3.2
    
    return x                                 # changed in ver 3.2 ' '.join(x.split())
##################################################################################


##################################################################################
#
#   Finds Index: Function to find index given subject and study numbers.
#
def getIndex (subject_arrays, study_array, subject, study):
    """Function to find index given subject and study numbers."""
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
def main():
    """Main entry point for the script."""

    start_time = time.time()

    splitData = pd.read_csv(SPLIT_FILE_NAME)       

# NegBio Data has label appied to all    
# Load, fill blanks with zero, convert all to intergers
    labelDataNeg = pd.read_csv(LABEL_FILE_NAME_NEG)
    labelDataNeg = labelDataNeg.fillna(0)
    labelDataNeg[list(labelDataNeg)] = labelDataNeg[list(labelDataNeg)].astype(int)
#    labelDataNeg['Findings'] = (labelDataNeg['No Finding'] != 1)*1

# ChesXpert Data has 2414 records with no labels - treat those as finding = 1    
# Load, fill blanks with zero, convert all to intergers
    labelDataChex = pd.read_csv(LABEL_FILE_NAME_CHEX)    
    labelDataChex = labelDataChex.fillna(0)    
    labelDataChex[list(labelDataChex)] = labelDataChex[list(labelDataChex)].astype(int)
#    labelDataChex['Findings'] = (labelDataChex['No Finding'] != 1)*1

    working_dir = os.getcwd()
    theOutFile = open(os.path.join(working_dir, OUT_FILE_NAME),"w",newline='')
    fileWriter = csv.writer(theOutFile, quoting=csv.QUOTE_MINIMAL)
    csvHeader = ['subject_id','study_id','search_term_code','split','neg_label','chex_label','reason_for_exam']
    fileWriter.writerow(csvHeader)
    for sCode in REASON_SEARCH_TERM_CODES:
        dirFileName = DIRECTORY_FILE_NAMES[sCode]
        dirFileTerm = REASON_SEARCH_STRINGS[sCode]
        theDirFile = open(os.path.join(working_dir, dirFileName),"r")
        for CurrentDirLine in theDirFile:
            CurrentParseFileName = CurrentDirLine[:CurrentDirLine.find(':')]
            theCurrentParseFile = open(CurrentParseFileName,"r")
            CurrentParseList = theCurrentParseFile.readlines()
            theCurrentParseFile.close()
            tList = []
            TermFound =False
            for s in CurrentParseList:
                tPos = s.strip().casefold().find(dirFileTerm.casefold())    # added ver 2.0 to handle 'CLINICAL INDICATION'
                if s.strip().casefold()[tPos:tPos+len(dirFileTerm)] == dirFileTerm.casefold():
                    # TermFound should be FALSE, but should add error check in case otherwise
#                    tList.append(s[tPos+len(dirFileTerm)+1:].strip().casefold())   # for no case version NCF & NCNF
                    tList.append(s[tPos+len(dirFileTerm)+1:].strip())               # for cased version CNF & CF                  
                    TermFound = True
                elif TermFound:
                    if s.strip() != '':
                        tList.append(s.strip())
                    else:
                        break
                else:
                    pass

            patientID = int(CurrentParseFileName[13:21])
            studyID = int(CurrentParseFileName[23:31])          #
            sReason = ' '.join(tList)                           #
            sReason = filter_reason(sReason)                   # uncomment for filtered version NCF & CF; comment-out for unfiltered version CNF & NCNF
            sReason = re.sub(r'[^a-zA-Z0-9]', ' ', sReason)     #
            sReason = ' '.join(sReason.split())                 #
            if sReason != '':  
                indx = getIndex(splitData['subject_id'],splitData['study_id'], patientID, studyID )
                writeRec = True
                if indx != -1:
                    theSplit = splitData.at[indx,'split']
                else:
                    print('Indx not found in splitdata file: subj={}, study={}.'.format(patientID,studyID))
                    writeRec = False
                    
                indx = getIndex(labelDataNeg['subject_id'],labelDataNeg['study_id'], patientID, studyID )
                if indx != -1:
                    labelNeg = 0 if labelDataNeg.at[indx,'No Finding'] else 1
                else:
                    print('Indx not found in labelDataNeg file: subj={}, study={}.'.format(patientID,studyID))
                    writeRec = False
    
                indx = getIndex(labelDataChex['subject_id'],labelDataChex['study_id'], patientID, studyID )
                if indx != -1:
                    labelChex = 0 if labelDataChex.at[indx,'No Finding'] else 1
                else:
                    print('Indx not found in findingChex file: subj={}, study={}.'.format(patientID,studyID))
                    writeRec = False
                
                if writeRec: 
                    fileWriter.writerow([patientID, studyID, sCode, theSplit, labelNeg, labelChex, sReason])
                else:
                    print('Skipping bad index record: patID {} study {}.'.format(patientID,studyID))
                
            else:
                print('Skipping empty string record: patID {} study {}.'.format(patientID,studyID))
                continue
        theDirFile.close()
    theOutFile.close()

    minutes, seconds = divmod((time.time() - start_time), 60)
    print('\nRun time:    --- {} minutes and {} seconds ---'.format(int(minutes), int(seconds)))

if __name__ == '__main__':
    sys.exit(main())


