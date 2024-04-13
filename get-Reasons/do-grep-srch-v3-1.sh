# !/bin/bash
 
# Performs GREP searchs of radiology reports.

grep -R ./files -e "INDICATION:" > grep-indication.txt
grep -R ./files -e "REASON FOR EXAM:" > grep-reasonForExam.txt
grep -R ./files -e "REASON FOR EXAMINATION:" > grep-reasonForExamination.txt
grep -R ./files -e "HISTORY:" > grep-history.txt
grep -R ./files -e "CLINICAL INFORMATION:" > grep-clinicalInfo.txt

