#!/bin/sh
# Runs full nBayes model
#
# 	where commandline arg 1 corresponds to REASON_FILTER (below)
#	and arg 2 is number of terms.
#
##
## [reason ver]       [reason filter]         [Num terms]
##     3-2x     ['NCF','NCNF','CNF','CF'] 	50, 200, 1500, ... 
##
declare -a rSet=(NCF NCNF CNF CF)

echo "Starting ${rSet[$1]}-$2 getBagWords  ..."
python3 do-nBayes/getBagWordLists-v2_1.py 3-2x $1 $2
echo "Starting ${rSet[$1]}-$2 getReasonVectors  ..."
python3 do-nBayes/getReasonVectorsTFIDF-v2_1.py 3-2x $1 $2
echo "Starting ${rSet[$1]}-$2 getSplits ..."
python3 do-nBayes/getSplitsXY-TFIDF-v2_1.py 3-2x $1 $2
echo "Starting ${rSet[$1]}-$2 doNaiveBayes ..."
python3 do-nBayes/doNaiveBayesTFIDF-v3_0.py 3-2x $1 $2 > NB-ALL-$2-${rSet[$1]}-v3-2x-randomSplits.out
echo "Done ${rSet[$1]}-$2."
