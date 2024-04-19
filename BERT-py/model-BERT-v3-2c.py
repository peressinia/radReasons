#!/usr/bin/env Python3
"""Script to use BERT models for binary classification of MIMIC-CXR radiology reports."""
#
# Version: 3.2c [15 Apr 24] based on v3.2
#
#   INPUT:  5 command line options
#               - string of reason file version
#               - integer index from 0-3 indicating reason file version
#               - integer index from 0-2 indicating the model
#               - integer index from 0-1 indicating split type (random, or MIMIC splits)
#               - integer > 0 for the number of epochs
#               - integer index from 0-7 indicating the learing rate
#
#           1 csv file with the results of the BERT model.
#
#   OUTPUT: a csv file with MIMIC splits and CheXpert labels for the vectorized reasons
#           
#           
#   EXAMPLE:    $ doNaiveBayesTFIDF-v3_1c 3-2x 0 50
#
#              The above looks for the reason file named:
#                   'mimic-reasons-NCF-v3-2x.csv'
#
#              It creates (or appends if it existed) the file:
#                   'model-BERT-Stat-Tally.csv' 
#              with the results of the BERT models
#

SCRIPT_VERSION = '3-2c'
SCRIPT_NAME = 'model-BERT'


import pandas as pd
import argparse, os.path
import time
from time import strftime
import datetime
import torch
import random
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score
from sklearn.model_selection import train_test_split
import seaborn as sns
from transformers import AutoConfig, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch.utils.data import TensorDataset
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, roc_curve, roc_auc_score
from torch.optim import AdamW
import os


#
# Set the seed values
#
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)           
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


# ================================================================= #
#  Function to calculate the accuracy of our predictions vs labels  #
# ================================================================= #
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
# ========================================




# ================================================================= #
#          Function to run the BERT trainig loop                    #
# ================================================================= #
def trainBERT (model, scheduler, total_steps, epochs, optimizer, batch_size,
               train_dataloader, validation_dataloader):

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []
    
    # Measure the total training time for the whole run.
    total_t0 = time.time()
    
    # For each epoch...
    for epoch_i in range(0, epochs):
    
        # ========================================
        #               Training
        # ========================================
    
        # Perform one full pass over the training set.
    
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
    
        # Measure how long the training epoch takes.
        t0 = time.time()
    
        # Reset the total loss for this epoch.
        total_train_loss = 0
    
        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()
    
        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
    
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
    
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
    
            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
    
            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()
    
            # Perform a forward pass (evaluate the model on this training batch).
            # In PyTorch, calling `model` will in turn call the model's `forward`
            # function and pass down the arguments. The `forward` function is
            # documented here:
            # https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
            # The results are returned in a results object, documented here:
            # https://huggingface.co/transformers/main_classes/output.html#transformers.modeling_outputs.SequenceClassifierOutput
            # Specifically, we'll get the loss (because we provided labels) and the
            # "logits"--the model outputs prior to activation.
            result = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask,
                           labels=b_labels,
                           return_dict=True)
    
            loss = result.loss
            logits = result.logits
    
            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()
    
            # Perform a backward pass to calculate the gradients.
            loss.backward()
    
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
    
            # Update the learning rate.
            scheduler.step()
    
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)
    
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)
    
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))
    
    
        # ========================================
        #               Validation
        # ========================================
       
        print("")
        print("Running Validation...")
        t0 = time.time()
        model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
    
        # Evaluate data for one epoch
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
    
            with torch.no_grad():
                result = model(b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask,
                               labels=b_labels,
                               return_dict=True)
    
            loss = result.loss
            logits = result.logits
    
            # Accumulate the validation loss.
            total_eval_loss += loss.item()
    
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
    
            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)
    
    
        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
    
        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)
    
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
    
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))
    
        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
    
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    total_train_time = time.time()-total_t0
    return training_stats, total_train_time

################### Train ##################################



###########################
def doTokenize(sentences, tokenizer):
###########################    
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            str(sent),                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = MAX_LENGTH_REASON,           # Pad & truncate all sentences.
                            truncation=True,
                            padding='max_length',
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    return input_ids, attention_masks

######################### doTokenize #############################










MAX_LENGTH_REASON = 64          # max number of tokens kept from the reason field

BATCH_SIZE_TRAINING = 32
BATCH_SIZE_TESTING  = 32
EPSILON = 1e-8                  
WARMUP_STEPS_PERCENT = 0        

TRAINING_PERCENT = 0.8
TEST_PERCENT = 0.1
VALIDATE_PERCENT = 0.1

# File to which performance measures are written
OUT_STAT_FILENAME = './model-BERT-Stat-Tally.csv'

LEARING_RATE_VERSIONS = [1e-4, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5] 
MODEL_VERSIONS = ['bert-base-uncased','UCSD-VA-health/RadBERT-RoBERTa-4m', 'emilyalsentzer/Bio_ClinicalBERT']
MODEL_VERSIONS_SHORT = ['baseBERT', 'radBERT', 'bioClinBERT']
REASON_FILTER_VERSIONS = ['NCF', 'NCNF', 'CNF','CF']
SPLIT_VERSIONS = ['mimic', 'random']

# ===============================================
# Function to process command line
# ===============================================
def doCL():
    """Parse the command line."""

    parser = argparse.ArgumentParser(description='Word-tokenize MIMIC-CXR radiology report INDICATIONS with frequencies.')
    parser.add_argument('reason_file_version', 
                    help='string representing version of reason file',
                    )
    parser.add_argument('reason_filter_index', help='integer from 0 to ' + str(len(REASON_FILTER_VERSIONS)) + ' indicating a filter type: ' + ''.join(str(REASON_FILTER_VERSIONS)),
                    type=int)
    parser.add_argument('model_index', help='integer from 0 to ' + str(len(MODEL_VERSIONS)) + ' indicating a filter type: ' + ''.join(str(MODEL_VERSIONS_SHORT)),
                    type=int)
    parser.add_argument('split_index', help='integer from 0 to ' + str(len(SPLIT_VERSIONS)) + ' indicating a split type: ' + ''.join(str(SPLIT_VERSIONS)),
                    type=int)
    parser.add_argument('num_epochs', help='positive integer for number of training epochs.'  ,
                    type=int)
    parser.add_argument('learning_rate', help='integer from 0 to ' + str(len(LEARING_RATE_VERSIONS)) + ' indicating a learning rate from: ' + ''.join(str(LEARING_RATE_VERSIONS)), 
                    type=int)
    args = parser.parse_args()
    return args.reason_file_version, args.reason_filter_index, args.model_index, args.split_index, args.num_epochs, args.learning_rate
# ===============================================






#######################################################################################################
##                                            Main Script                                            ##
#######################################################################################################
start_time = time.time()
reason_file_version, reason_filter, model_index, split_index, num_epochs, learning_rate_index = doCL()
TAG_STR = REASON_FILTER_VERSIONS[reason_filter] + '-v' + reason_file_version
learning_rate = LEARING_RATE_VERSIONS[learning_rate_index]
lr_str = "{:.0e}".format(learning_rate).replace('-','')
output_header = (MODEL_VERSIONS_SHORT[model_index] + '-' + SPLIT_VERSIONS[split_index] + 
  'splits' + '-ep' + str(num_epochs) + 'bSz' + str(BATCH_SIZE_TRAINING) + 'nTok' + 
  str(MAX_LENGTH_REASON) + 'wup' + str(WARMUP_STEPS_PERCENT) + 'lr' + lr_str + 'data' +  
  TAG_STR + 'ScrptV' + SCRIPT_VERSION)
reason_file_name = './get-reasons/mimic-reasons-' + REASON_FILTER_VERSIONS[reason_filter] + '-v' + reason_file_version + '.csv'
dataSaveDF = pd.DataFrame(columns = ['verTag', 'model', 'reason-filter', 'reason-version',  'splits', 
                                     'epochs', 'learning-rate', 'warmup-steps', 'batch-size',
                                     'nTrain', 'nValidate', 'nTest',
                                     'accTest', 'MCC', 'RocAuc', 'f1-0', 'f1-1',
                                     'num-0', 'num-1', 'precision', 'recall', 'tp-rate', 'fp-rate', 'specificity',
                                     'decile1-n', 'd1-pmin', 'd1-pmax', 'lift1', 'ks-stat', 'ks-decile',
                                     'TP', 'TN', 'FP', 'FN', 'timestamp'])
dataSave = [output_header, MODEL_VERSIONS_SHORT[model_index], REASON_FILTER_VERSIONS[reason_filter], reason_file_version, 
            SPLIT_VERSIONS[split_index] + '' if split_index == 0 else 'random:'+ 
            str(int(TRAINING_PERCENT*100)) + '-' + str(int(VALIDATE_PERCENT*100)) + '-' + str(int(TEST_PERCENT*100)),
            num_epochs, learning_rate, WARMUP_STEPS_PERCENT, BATCH_SIZE_TRAINING]

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
theData = pd.read_csv(reason_file_name)
print('reason datafile: {}'.format(reason_file_name))
print('Total number of reasons: {:,}\n'.format(theData.shape[0]))
sentences = theData.reason_for_exam.values
labels = theData.chex_label.values
splits = theData.split.values

# Create a train-validation-test split.
if SPLIT_VERSIONS[split_index] == 'mimic':
    sentences_train = sentences[splits == 'train' ]
    sentences_validate = sentences[splits == 'validate' ]
    sentences_test = sentences[splits == 'test' ]
    labels_train = labels[splits == 'train' ]
    labels_validate = labels[splits == 'validate' ]
    labels_test = labels[splits == 'test' ]
    train_size = len(sentences_train)
    validate_size = len(sentences_validate)
    test_size = len(sentences_test)   
else:
    train_size = round(TRAINING_PERCENT * len(theData))
    validate_size = round(VALIDATE_PERCENT * len(theData))
    test_size = len(theData) - (train_size + validate_size)
    sentences_train, sentences_test, labels_train, labels_test = train_test_split(
        sentences, labels, test_size=(test_size+validate_size), random_state=42)
    sentences_test, sentences_validate, labels_test, labels_validate  = train_test_split(
        sentences_test, labels_test, test_size=validate_size, random_state=42)


print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(validate_size))
print('{:>5,} test samples'.format(test_size))
dataSave.extend([train_size, validate_size, test_size])

#######################################################################
#                      Load Model & Tokenizer                         #
#######################################################################
print('Loading ' + MODEL_VERSIONS_SHORT[model_index] + ' model...')
config = AutoConfig.from_pretrained(MODEL_VERSIONS[model_index])
tokenizer = AutoTokenizer.from_pretrained(MODEL_VERSIONS[model_index])



#######################################################################
#               Tokenize, and create tensors                          #
#######################################################################
input_ids, attention_masks = doTokenize(sentences_train,tokenizer)
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels_train)
# Combine the training inputs into a TensorDataset.
train_dataset = TensorDataset(input_ids, attention_masks, labels)

input_ids, attention_masks = doTokenize(sentences_test,tokenizer)
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels_test)
# Combine the training inputs into a TensorDataset.
test_dataset = TensorDataset(input_ids, attention_masks, labels)

input_ids, attention_masks = doTokenize(sentences_validate,tokenizer)
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels_validate)
# Combine the training inputs into a TensorDataset.
validation_dataset = TensorDataset(input_ids, attention_masks, labels)



#######################################################################
#                     Setup Model and Train                           #
#######################################################################

batch_size = BATCH_SIZE_TRAINING
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )
validation_dataloader = DataLoader(
            validation_dataset, # The validation samples.
            sampler = SequentialSampler(validation_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )


# Initialize the radBERT model.
print('Initializing ' + MODEL_VERSIONS_SHORT[model_index] + ' model...')

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_VERSIONS[model_index], 
    num_labels = 2, 
    output_attentions = False,
    output_hidden_states = False,
)

# Tell pytorch to run this model on the GPU.
model.cuda()
optimizer = AdamW(model.parameters(),
                  lr = learning_rate,
                  eps = EPSILON
                )
epochs = num_epochs

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
wUp = 0 if WARMUP_STEPS_PERCENT == 0 else round(WARMUP_STEPS_PERCENT*total_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = wUp, 
                                            num_training_steps = total_steps)



training_stats, total_train_time = trainBERT(model, scheduler, total_steps, epochs, 
                                             optimizer, batch_size, train_dataloader, validation_dataloader)





#######################################################################
#                 Output Trainings Stats and Graphs                   #
#######################################################################

# Display floats with two decimal places.
pd.set_option('display.precision', 2)
df_stats = pd.DataFrame(data=training_stats)
df_stats = df_stats.set_index('epoch')
print(df_stats)

# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)

# Plot the learning curve.
plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

# Label the plot.


plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
#plt.figtext(0.5, 0.01, plotDetails, wrap=True, horizontalalignment='center', fontsize=12)
plt.ylabel("Loss")
plt.legend()
plt.xticks([i+1 for i in range(num_epochs)])
plt.savefig('tvLoss-{}.png'.format(output_header)) 
plt.clf() # this clears the figure




#######################################################################
#            Setup and run predictions on test data set.              #
#######################################################################

# Set the batch size.
batch_size =BATCH_SIZE_TESTING


# Create the DataLoader.
prediction_data = test_dataset         
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









#######################################################################
#                Evaluate Performance on Test data.                   #
#######################################################################

# The predictions are a 2-column ndarray (one column for "0"
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


# Evaluate each test batch using Matthew's correlation coefficient
print('Calculating Matthews Corr. Coef. for each batch...')

# Calculate the MCC
mcc = matthews_corrcoef(flat_true_labels, flat_predictions)

print('\n===================================================')
print('\nMaxLen={}, batch={}, epochs={}, warmup percent={}, learing rate={}.'.format(MAX_LENGTH_REASON,BATCH_SIZE_TRAINING,num_epochs,WARMUP_STEPS_PERCENT,learning_rate))
print('Model: {}\n'.format(MODEL_VERSIONS[model_index]))
print('Datafile: {}\n'.format(reason_file_name))
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
print('Model accuracy  score: {0:0.4f}'.format(ac))
print('Model precision score: {0:0.4f}'.format( precision_score(y_test, y_pred)))
cm_matrix = pd.DataFrame(data=cm, index=['Actual Negative:0', 'Actual Positive:1'], columns=['Predict Negative:0', 'Predict Positive:1'])                                 


dataSave.extend([ac,mcc])     

plt1 = sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
theFig = plt1.get_figure()
output_file = './ConfusionMatrix-' + output_header
theFig.savefig(output_file) 
theFig.clf() # this clears the figure

RocAuc = roc_auc_score(y_test,y_pred)
print('ROC_AUC Score: {0:0.4f}'.format(RocAuc))
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.figure(figsize=(7,7))
output_file = './ROC-Plot-' + output_header
Roc_ti_String = MODEL_VERSIONS_SHORT[model_index] + ' with ' + REASON_FILTER_VERSIONS[reason_filter] + ' data (ep=' + str(num_epochs) +')'
from plot_metric.functions import BinaryClassification # Visualisation with plot_metric
bc = BinaryClassification(y_test, y_pred, labels=["Class 1", "Class 2"])
bc.plot_roc_curve(plot_threshold=False,title='ROC Curve: ' + Roc_ti_String)
plt.savefig(output_file)
#plt.show()
plt.clf()


cRep = classification_report(y_test, y_pred,output_dict=True)
dataSave.extend([RocAuc,cRep['0']['f1-score'],cRep['1']['f1-score'],cRep['0']['support'],cRep['1']['support']])
print(classification_report(y_test, y_pred,output_dict=False))

TN, FP, FN, TP = cm.ravel()

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

# Save stats for output later
dataSave.extend([precision,recall,true_positive_rate,false_positive_rate,specificity])
#







#######################################################################
#                Evaluate Lift                                        #
#######################################################################

import KDS

y_prob = np.max(flat_probs, axis=1)

xx=KDS.decile_table(flat_true_labels, y_prob, labels=False)
ksmx = xx.KS.max()
ksdcl = xx[xx.KS == ksmx].decile.values    
print('===========================================================')
decToPrint = 0
print('Decile {}: count = {}, prob min = {}, prob max = {}.'.format(xx.iloc[decToPrint,0],
                                                                    xx.iloc[decToPrint,5],
                                                                    xx.iloc[decToPrint,1],
                                                                    xx.iloc[decToPrint,2]))
print('lift @ decile {} = {}.'.format(decToPrint,xx['lift'].iloc[0]))
print('KS Stat: {} @ decile {}.'.format(str(ksmx),str(list(ksdcl)[0])))
print('===========================================================')


# Save stats for output later
dataSave.extend([xx.iloc[decToPrint,5],xx.iloc[decToPrint,1],xx.iloc[decToPrint,2],xx['lift'].iloc[0],ksmx,ksdcl[0]])



print('===========================================================')        
minutes, seconds = divmod((time.time() - start_time), 60)
print('\nRun time:    --- {} minutes and {} seconds ---'.format(int(minutes), int(seconds)))
print('===========================================================')

# Save stats for output later
dataSave.extend([TP,TN,FP,FN])
dataSave.extend([strftime("%d %b %Y %H:%M:%S", time.localtime(time.time()))])
dataSaveDF.loc[-1] = dataSave            # add row
dataSaveDF.index = dataSaveDF.index + 1  # shifting index
dataSaveDF = dataSaveDF.sort_index()     # sorting by index

###### Output accuracy stats ######
if(os.path.isfile(OUT_STAT_FILENAME)):
    dataSaveDF.to_csv(OUT_STAT_FILENAME, encoding='utf-8', mode='a', index=False, header=False)  #  file exists, append
else:   
    dataSaveDF.to_csv(OUT_STAT_FILENAME, encoding='utf-8', index=False, header=True)             # create file with headers








#######################################################################
#                          Save the model.                            #
#######################################################################

output_file = './_model_save-' + output_header 
print('Saving model to: {}'.format(output_file))

model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_file)
tokenizer.save_pretrained(output_file)




minutes, seconds = divmod((time.time() - start_time), 60)
print('\nRun time:    --- {} minutes and {} seconds ---'.format(int(minutes), int(seconds)))


