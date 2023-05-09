#!/usr/bin/env Python3
"""Script to use plain BERT for binary classification of MIMIC-CXR radiology reports."""
#
# Script:   loadModel-BERT.py
#                       version: 1.0  [based on model-BERT.py ver 2.0 ]
#                       
#
#
#
#
SCRIPT_VERSION = '1-0'


MAX_LENGTH_REASON = 64          # max number of tokens kept from the reason field
EPOCHS_TRAINING = 10
BATCH_SIZE_TRAINING = 32
BATCH_SIZE_TESTING  = 32
LEARNING_RATE = 2e-5            # args.learning_rate - default is 5e-5, I've used 2e-5
LR_STR = '2e5'
EPSILON = 1e-8                  # args.adam_epsilon  - default is 1e-8.
WARMUP_STEPS_PERCENT = 0        # used 0 so far

TRAINING_PERCENT = 0.8
VAL_TEST_RATIO = 0.5  # Split what's left after Training split by this VAL:TEST ration


## Version 6 is uncased
DATAFILE_NAME = 'mimic-reasons-BERT-v6.csv'
DATAFILE_VER = 'V6'

## Version 7 is cased
# DATAFILE_NAME = 'mimic-reasons-BERT-v7.csv'
# DATAFILE_VER = 'V7'

MODEL_FILENAME = '_model_save-bioClinBERT-ep8bSz32nTok64wup0lr3e5dataV6ScrptV2-0'
output_header = 'bioClinBERT-ep8bSz32nTok64wup0lr3e5dataV6ScrptV2-0'

#MODEL_NAME_STR = 'bert-base-uncased'
#MODEL_NAME_STR_SHORT = 'baseBERT'


# MODEL_NAME_STR = 'bert-large-uncased'
# MODEL_NAME_STR_SHORT = 'largeBERT'

#MODEL_NAME_STR = 'roberta-base'
#MODEL_NAME_STR_SHORT = 'roBERTa'

# MODEL_NAME_STR = 'UCSD-VA-health/RadBERT-RoBERTa-4m'
# MODEL_NAME_STR_SHORT = 'radBERT'

MODEL_NAME_STR = 'emilyalsentzer/Bio_ClinicalBERT'
MODEL_NAME_STR_SHORT = 'bioClinBERT'


import pandas as pd
import time
import torch
import random
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# from sklearn.model_selection import train_test_split
import seaborn as sns
from transformers import AutoConfig, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch.utils.data import TensorDataset, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader, SequentialSampler


# Set the seed value all over the place to make this reproducible.
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)            # used by Sklearn
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)




# ========================================
# Function to calculate the accuracy of our predictions vs labels
# ========================================
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
# ========================================



###########################
def doTokenize(sentences, tokenizer):
###########################    
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            str(sent),                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = MAX_LENGTH_REASON,           # Pad & truncate all sentences.
                            truncation=True,
                            padding='max_length',
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )
        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
    return input_ids, attention_masks

######################### doTokenize #############################










#######################################################################################################
##                                            Main Script                                            ##
#######################################################################################################
start_time = time.time()



#######################################################################
#                        Check for GPU                                #
#######################################################################
if torch.cuda.is_available():
    # Assign GPU to PyTorch.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")






#######################################################################
#                           Get data                                  #
#######################################################################
theData = pd.read_csv(DATAFILE_NAME)
print('datafile: {}'.format(DATAFILE_NAME))
print('Total number of reasons: {:,}\n'.format(theData.shape[0]))
sentences = theData.reason_for_exam.values
labels = theData.chex_label.values

# sentences_train, sentences_rem, labels_train, labels_rem = train_test_split(sentences,labels, train_size=TRAINING_PERCENT)
# sentences_validate, sentences_test, labels_validate, labels_test = train_test_split(sentences_rem,labels_rem, train_size=VAL_TEST_PERCENT)


    


#######################################################################
#                      Load Model & Tokenizer                         #
#######################################################################
print('Loading ' + MODEL_NAME_STR_SHORT + ' model...')
input_dir = './' + MODEL_FILENAME
model = AutoModelForSequenceClassification.from_pretrained(input_dir)
tokenizer = AutoTokenizer.from_pretrained(input_dir)

# Copy the model to the GPU.
model.to(device)

# Tell pytorch to run this model on the GPU.
#model.cuda()



#######################################################################
#               Tokenize, and create tensors                          #
#######################################################################
input_ids, attention_masks = doTokenize(sentences,tokenizer)
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks, labels)

# Create a train-validation-test split.
train_size = int(TRAINING_PERCENT * len(dataset))
rem_size = len(dataset) - train_size
train_dataset, rem_dataset = random_split(dataset, [train_size, rem_size])
validate_size = int(VAL_TEST_RATIO * len(rem_dataset))
test_size = len(rem_dataset) - validate_size
validation_dataset, test_dataset = random_split(dataset, [train_size, rem_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(validate_size))
print('{:>5,} test samples'.format(test_size))









#######################################################################
#            Setup and run predictions on test data set.              #
#######################################################################

# Set the batch size.
batch_size =BATCH_SIZE_TESTING



# Create the DataLoader.
prediction_data = test_dataset          #TensorDataset(input_ids, attention_masks, labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

# Prediction on test set
print('\nPredicting labels for {:,} test sentences...'.format(test_size))
# Put model in evaluation mode
model.eval()
# Tracking variables
predictions , true_labels, probabilities = [], [], []


import torch.nn.functional as F
# Predict
for batch in prediction_dataloader:
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)

  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_labels = batch

  # Telling the model not to compute or store gradients, saving memory and
  # speeding up prediction
  with torch.no_grad():
      # Forward pass, calculate logit predictions.
      result = model(b_input_ids,
                     token_type_ids=None,
                     attention_mask=b_input_mask,
                     return_dict=True)

  logits = result.logits
  logitsX = result.logits
  probX = F.softmax(logitsX, dim=-1)
  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()
  probX = probX.detach().cpu().numpy()

  # Store predictions and true labels
  predictions.append(logits)
  true_labels.append(label_ids)
  probabilities.append(probX)
  
print('    DONE.\n')





#logitsX = result.logits
#probabilities = F.softmax(logitsX, dim=-1)




#######################################################################
#                Evaluate Performance on Test data.                   #
#######################################################################

# The predictions and probs are a 2-column ndarray (one column for "0"
# and one column for "1"). Pick the label with the highest value and turn this
# in to a list of 0s and 1s.

# Combine the predictions for each batch into a single list. 
flat_predictions = np.concatenate(predictions, axis=0)
# For each sample, pick the label (0 or 1) with the higher score.
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
# Combine the correct labels for each batch into a single list.
flat_true_labels = np.concatenate(true_labels, axis=0)

flat_probs = np.concatenate(probabilities, axis=0)
flat_probs_preds = np.argmax(flat_probs, axis=1).flatten()  # should be the same as flat_predictions

print('Positive samples: %d of %d (%.2f%%)' % (sum(flat_true_labels), len(flat_true_labels), (sum(flat_true_labels) / len(flat_true_labels) * 100.0)))

matthews_set = []

# Evaluate each test batch using Matthew's correlation coefficient
print('Calculating Matthews Corr. Coef. for each batch...')

# Calculate the MCC
mcc = matthews_corrcoef(flat_true_labels, flat_predictions)

print('\n===================================================')
print('\nMaxLen={}, batch={}, epochs={}, warmup percent={}, learing rate={}.'.format(MAX_LENGTH_REASON,BATCH_SIZE_TRAINING,EPOCHS_TRAINING,WARMUP_STEPS_PERCENT,LEARNING_RATE))
print('Model: {}\n'.format(MODEL_NAME_STR))
print('Datafile: {}\n'.format(DATAFILE_NAME))
print('===================================================')
print('Total MCC: %.3f' % mcc)
print('flat Pred: {}.'.format(flat_predictions))
print('flat True: {}.'.format(flat_true_labels))
theAccuracy = np.sum(flat_predictions == flat_true_labels) / len(flat_true_labels)
print('\nAccuracy = {}'.format(theAccuracy))
print('===================================================\n\n')

print('===========================================================')
y_test = flat_true_labels
y_pred = flat_predictions
ac = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred)
print('\ncm = {}\n'.format(cm))
print('Model accuracy score: {0:0.4f}'.format(ac))
cm_matrix = pd.DataFrame(data=cm, index=['Actual Negative:0', 'Actual Positive:1'], columns=['Predict Negative:0', 'Predict Positive:1'])                                 
#plt1 = sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
#theFig = plt1.get_figure()
#output_file = './ConfusionMatrix-' + output_header

#theFig.savefig(output_file) 
#theFig.clf() # this clears the figure

print(classification_report(y_test, y_pred))
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]
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
print('===========================================================\n\n')








#######################################################################
#                Evaluate Lift                                        #
#######################################################################



y_prob = np.max(flat_probs, axis=1)
#X_test['predicted_prob'] = y_prob[:,1]

import myKDS as mK

# CUMMULATIVE GAIN PLOT
theFig = mK.plot_cumulative_gain(flat_true_labels , y_prob)  
theFig.savefig('fig-CumGain-{}.pdf'.format(output_header)) 
theFig.clf() # this clears the figure

# LIFT PLOT
theFig = mK.plot_lift(flat_true_labels , y_prob)  
theFig.savefig('fig-lift-{}.pdf'.format(output_header)) 
theFig.clf() # this clears the figure

# KS Stat PLOT
theFig = mK.plot_ks_statistic(flat_true_labels , y_prob)  
theFig.savefig('fig-ksStat-{}.pdf'.format(output_header)) 
theFig.clf() # this clears the figure

dt, theFig = mK.report(flat_true_labels, y_prob,plot_style='ggplot')
#theFig.savefig('fig-4lift-{}.pdf'.format(output_header)) 
#theFig.clf() # this clears the figure


xx=mK.decile_table(flat_true_labels, y_prob)    
print('===========================================================')
decToPrint = 0
print('Decile {}: count = {}, prob min = {}, prob max = {}.'.format(xx.iloc[decToPrint,0],
                                                                    xx.iloc[decToPrint,5],
                                                                    xx.iloc[decToPrint,1],
                                                                    xx.iloc[decToPrint,2]))
print('lift @ decile {} = {}.'.format(decToPrint,xx['lift'].iloc[0]))
print('===========================================================')




#plot_cumulative_gain(flat_true_labels, flat_true_labels, title='Cumulative Gain Plot', title_fontsize=14, text_fontsize=10, figsize=None)






###### Output time stats ######

minutes, seconds = divmod((time.time() - start_time), 60)
print('\nRun time:    --- {} minutes and {} seconds ---'.format(int(minutes), int(seconds)))


