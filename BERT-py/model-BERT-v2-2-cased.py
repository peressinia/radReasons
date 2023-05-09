#!/usr/bin/env Python3
"""Script to use plain BERT for binary classification of MIMIC-CXR radiology reports."""
#
# Script:   model-BERT-cased.py [added case options in tokenizer ]
#
#                       version: 2.2  [fixed test set size error]                       
#                       version: 2.1  [added lift study]
#                       version: 2.0  [cleaned, streamlined, restructured, parameterized further
#                       version: 1.4  [incorporates data version for cased models.]
#                       version: 1.3  [added bert-large-uncased & roBERTa;  cleaned flag warnings;
#                                      combined doTokenize function; confusion & stats] 
#                       version: 1.2  [fix radBERT sticking problem] 
#                       version: 1.1  [uses validation set for validation in training epochs]
#
#
#
#
SCRIPT_VERSION = '2-2-C'


MAX_LENGTH_REASON = 64          # max number of tokens kept from the reason field
EPOCHS_TRAINING = 8
BATCH_SIZE_TRAINING = 32
BATCH_SIZE_TESTING  = 32
LEARNING_RATE = 3e-5            # args.learning_rate - default is 5e-5, I've used 2e-5
LR_STR ='3e5'
EPSILON = 1e-8                  # args.adam_epsilon  - default is 1e-8.
WARMUP_STEPS_PERCENT = 0        # used 0 so far

TRAINING_PERCENT = 0.8
VAL_TEST_RATIO = 0.5  # Split what's left after Training split by this VAL:TEST ration


## Version 6 is uncased
# DATAFILE_NAME = 'mimic-reasons-BERT-v6.csv'
# DATAFILE_VER = 'V6'

## Version 7 is cased
DATAFILE_NAME = 'mimic-reasons-BERT-v7.csv'
DATAFILE_VER = 'V7'

# MODEL_NAME_STR = 'bert-base-uncased'
# MODEL_NAME_STR_SHORT = 'baseBERT'

# MODEL_NAME_STR = 'bert-large-uncased'
# MODEL_NAME_STR_SHORT = 'largeBERT'


#MODEL_NAME_STR = 'roberta-base'
#MODEL_NAME_STR_SHORT = 'roBERTa'


#MODEL_NAME_STR = 'UCSD-VA-health/RadBERT-RoBERTa-4m'
#MODEL_NAME_STR_SHORT = 'radBERT'

MODEL_NAME_STR = 'emilyalsentzer/Bio_ClinicalBERT'
MODEL_NAME_STR_SHORT = 'bioClinBERT'


import pandas as pd
import time
import datetime
import torch
import random
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# from sklearn.model_selection import train_test_split
import seaborn as sns
from transformers import AutoConfig, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch.utils.data import TensorDataset, random_split
#from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
from torch.optim import AdamW
import os



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
        # After the completion of each training epoch, measure our performance on
        # our validation set.
    
        print("")
        print("Running Validation...")
    
        t0 = time.time()
    
        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()
    
        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
#        nb_eval_steps = 0
    
        # Evaluate data for one epoch
        for batch in validation_dataloader:
    
            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
    
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
    
                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                result = model(b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask,
                               labels=b_labels,
                               return_dict=True)
    
            # Get the loss and "logits" output by the model. The "logits" are the
            # output values prior to applying an activation function like the
            # softmax.
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
                            return_tensors = 'pt'     # Return pytorch tensors.                            
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
output_header = MODEL_NAME_STR_SHORT + '-ep' + str(EPOCHS_TRAINING) + 'bSz' + str(BATCH_SIZE_TRAINING) + 'nTok' + str(MAX_LENGTH_REASON) + 'wup' + str(WARMUP_STEPS_PERCENT) + 'lr' + LR_STR + 'data' + DATAFILE_VER + 'ScrptV' + SCRIPT_VERSION


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
#         Get tokenizer, tokenize, and create tensors                 #
#######################################################################
config = AutoConfig.from_pretrained(MODEL_NAME_STR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_STR,do_lower_case=False)  # case added ver 2.1
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
validation_dataset, test_dataset = random_split(rem_dataset, [validate_size, test_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(validate_size))
print('{:>5,} test samples'.format(test_size))







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
print('Initializing ' + MODEL_NAME_STR_SHORT + ' model...')

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME_STR, # the particular BERT model (?uncased vocab?)
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

# Tell pytorch to run this model on the GPU.
model.cuda()

optimizer = AdamW(model.parameters(),
                  lr = LEARNING_RATE,
                  eps = EPSILON
                )
epochs = EPOCHS_TRAINING

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
plt.xticks([i+1 for i in range(EPOCHS_TRAINING)])
plt.savefig('tvLoss-{}.png'.format(output_header)) 
plt.clf() # this clears the figure




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

matthews_set = []

# Evaluate each test batch using Matthew's correlation coefficient
print('Calculating Matthews Corr. Coef. for each batch...')

# Calculate the MCC
mcc = matthews_corrcoef(flat_true_labels, flat_predictions)

print('\n===================================================')
print('\nMaxLen={}, batch={}, epochs={}, warmup percent={}, learing rate={}.'.format(MAX_LENGTH_REASON,BATCH_SIZE_TRAINING,EPOCHS_TRAINING,WARMUP_STEPS_PERCENT,LEARNING_RATE))
print('Model: {}\n'.format(MODEL_NAME_STR))
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
plt1 = sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
theFig = plt1.get_figure()
output_file = './ConfusionMatrix-' + output_header

theFig.savefig(output_file) 
theFig.clf() # this clears the figure

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








#######################################################################
#                          Save the model.                            #
#######################################################################

# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

# output_dir = './_model_save-radBert-epoch4batch32/'
output_file = './_model_save-' + output_header 
print('Saving model to: {}'.format(output_file))

# Create output directory if needed
if not os.path.exists(output_file):
    os.makedirs(output_file)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_file)
tokenizer.save_pretrained(output_file)

# Good practice: save your training arguments together with the trained model
# torch.save(args, os.path.join(output_dir, 'training_args.bin'))







minutes, seconds = divmod((time.time() - start_time), 60)
print('\nRun time:    --- {} minutes and {} seconds ---'.format(int(minutes), int(seconds)))

