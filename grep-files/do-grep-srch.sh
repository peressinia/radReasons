# !/bin/bash
 
# Performs GREP searchs of radiology reports.

grep -Ri ./files -e "indication:" > grep-indication.txt
grep -Ri ./files -e "reason for exam:" > grep-reasonForExam.txt
grep -Ri ./files -e "reason for examination:" > grep-reasonForExamination.txt

