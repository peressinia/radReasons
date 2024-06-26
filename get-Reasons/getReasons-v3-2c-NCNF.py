#!/usr/bin/env Python3
"""Script to extract MIMIC-CXR radiology report REASON FOR EXAM field with NCNF filter."""
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
#   OUTPUT: a non-cased, non-filtered Indications csv file with column headers as follows:
#           [subject_id, study_id, search_term_code, split, neg_label, chex_label, reason_for_exam]
#           
#           
#   EXAMPLE:    $ getReasons-v3-2-NCNF
#
#              The above looks for the 8 files list above ...
#
#              It creates the file of reasons:
#                   'mimic-reasons-NCNF-v3-2x.csv'
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
OUT_FILE_NAME = 'mimic-reasons-NCNF-v3-2.csv'

LABEL_FILE_NAME_NEG = './mimic-cxr-2.0.0-negbio.csv'
LABEL_FILE_NAME_CHEX = './mimic-cxr-2.0.0-chexpert.csv'
SPLIT_FILE_NAME = './mimic-cxr-2.0.0-split.csv'


#############################################################
def filter_reason(x):
    """Function to filter and replace abbreviation in REASON FOR EXAM string."""
    x = x.casefold()

# Next Set replace abbviations for sex and standardize
    x = x.replace('___f ', ' female ')
    x = x.replace('___/f ', ' female ')
    x = x.replace('___ f ', ' female ')
    x = x.replace('___ yo f ', ' female ')
    x = x.replace('___ yr f ', ' female ')
    x = x.replace('___ y f ', ' female ')
    x = x.replace('___ y/o f ', ' female ')
    x = x.replace('___ year f ', ' female ')
    x = x.replace('___-year old f ', ' female ')
    x = x.replace('___-year-old f ', ' female ')
    x = x.replace('___ year old f ', ' female ')
    x = x.replace('___ year-old f ', ' female ')
    x = x.replace('___ years old f ', ' female ')

    x = x.replace('___ y.o f ', ' female ')
    x = x.replace('___ y.o. f ', ' female ')
    x = x.replace(' ___ f ', ' female ')
    x = x.replace('y/o f ', ' female ')
    x = x.replace('woman', ' female ')

    x = x.replace('___m ', ' male ')
    x = x.replace('___/m ', ' male ')
    x = x.replace('___ m ', ' male ')
    x = x.replace('___ yo m ', ' male ')
    x = x.replace('___ yr m ', ' male ')
    x = x.replace('___ y m ', ' male ')
    x = x.replace('___ y/o m ', ' male ')
    x = x.replace('___ year m ', ' male ')
    x = x.replace('___-year old m ', ' male ')
    x = x.replace('___-year-old m ', ' male ')
    x = x.replace('___ year old m ', ' male ')
    x = x.replace('___ year-old m ', ' male ')
    x = x.replace('___ years old m ', ' male ')

    x = x.replace('___ y.o m ', ' male ')
    x = x.replace('___ y.o. m ', ' male ')
    x = x.replace(' ___ m ', ' male ')
    x = x.replace('y/o m ', ' male ')
    x = x.replace('man', ' male ')
    
    x = x.replace('yo ', ' ')               # ver 2.0
    x = x.replace('y o ', ' ')              # ver 2.0


# Next Set is for other known abbriviations
    x = x.replace('s/p', ' status post ')  #ver 2.0
    x = x.replace(' sp ', ' status post ')  #ver 2.0
    x = x.replace('s p ', ' status post ')  #ver 2.0

    x = x.replace('c/o', ' complains of ')  #ver 1.1 pullReason
    x = x.replace(' w/', ' with ')  #ver 1.1 pullReason

    x = x.replace('r/o', ' rule_out ')
    x = x.replace(' ro ', ' rule_out ')
    x = x.replace('rule out', ' rule_out ')

    x = x.replace('f/u', ' follow_up ')
    x = x.replace('follow up', ' follow_up ')

    x = x.replace('ptx', ' pneumothorax ')

    x = x.replace('oxygen requirement', ' oxygen_requirement ')
    x = x.replace('o2 requirement', ' oxygen_requirement ')
    x = x.replace('o2', 'oxygen')
    
    x = x.replace('questionable', 'question')
    x = x.replace('confirming', 'confirm')

    x = x.replace('chf', ' cong_heart_failure ')
    x = x.replace('congestive heart failure', ' cong_heart_failure ')

    x = x.replace('sob', ' shortness_of_breath ')
    x = x.replace('shortness of breath', ' shortness_of_breath ')
    x = x.replace('pulmonary', ' pulm ')

    x = x.replace('endotracheal tube', ' endotracheal_tube ')
    x = x.replace('et tube', ' endotracheal_tube ')
    x = x.replace('ett tube', ' endotracheal_tube ')

    x = x.replace(' ca ', ' cancer ')
    x = x.replace(' ca.', ' cancer ')    

#    x = x.replace('coronary artery disease', ' cad ')
    x = x.replace(' cad ', 'coronary artery disease')

    x = x.replace('fxs', ' fracture ')
    x = x.replace('fx', ' fracture ')
    x = x.replace('fractures', ' fracture ')

#    x = x.replace('end-stage renal disease', ' esrd ') 
#    x = x.replace('end stage renal disease', ' esrd ')     
    x = x.replace(' esrd ', ' end stage renal disease ') 

    x = x.replace('fluid overload', ' fluid_overload ')
    x = x.replace('volume overload', ' volume_overload ')    

    x = x.replace('htn', ' hypertension ')

    x = x.replace('pulmonary embolism', ' pe ')

    x = x.replace('pna', ' pneumonia ')  # space before "pna" removed v. 1.2

    x = x.replace('nasogastric tube', ' nasogastric_tube ')
    x = x.replace('ngt', ' nasogastric_tube ')
    x = x.replace('ng tube', ' nasogastric_tube ')

#    x = x.replace('right lower lobe', ' rll ')        
    x = x.replace('rll',' right lower lobe ')        

    x = x.replace('avr', ' aortic valve_replacement ')        
    x = x.replace('mvr', ' mitral valve_replacement ')        
    x = x.replace('tvr', ' tricuspid valve_replacement ')        
    x = x.replace('valve replacement', ' valve_replacement ')        


    x = x.replace('___', ' ')
    x = x.replace('year', ' ')    # ver 2.0
    x = x.replace('old', ' ')     # ver 2.0
    
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
                    tList.append(s[tPos+len(dirFileTerm)+1:].strip().casefold())    # for no case version NCF & NCNF
#                    tList.append(s[tPos+len(dirFileTerm)+1:].strip())              # for cased version CNF                   
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
#            sReason = filter_reason(sReason)                   # uncomment for filtered version NCF; comment-out for unfiltered version CNF & NCNF
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


