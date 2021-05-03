# deeper cnn model for GH13 family
# Imports
#=====================#

from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from keras.utils import to_categorical
from keras.models import Sequential
#from keras.layers import Conv2D
#from keras.layers import MaxPooling2D
#from keras.optimizers import SGD
from keras.optimizers import Adam

import pandas as pd
import numpy as np
from scipy import stats
import random
from Bio import SeqIO
import os
import subprocess

from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import bioinformatics as bioinf

from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers.merge import Concatenate
from keras.layers import Dense,Dot,Conv1D,MaxPooling1D,Activation,Dropout,LSTM,Flatten,GlobalMaxPooling1D,Input,BatchNormalization
import pydot
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras.callbacks import ReduceLROnPlateau
import os

import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


# load train and test dataset
heads, sequences = bioinf.split_fasta('fasta/GH13_positions_only/GH13_cat.fasta')
subtype = list(pd.read_csv('results_final/ncbi_subtypes.csv')['ncbi_pred_class'])
lb = LabelBinarizer()
y = lb.fit_transform(subtype)
y = to_categorical(y)
cat_domain_fasta = 'fasta/GH13_positions_only/GH13_cat.fasta'
sequence_df = bioinf.fasta_to_df(cat_domain_fasta)
max_length = len(sequence_df.columns)
embedding_dim = 11
top_classes=2
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(sequences)

X_seq = tokenizer.texts_to_sequences(sequences)
X_seq = sequence.pad_sequences(X_seq, maxlen=max_length)
#X_train, X_test, y_train, y_test = train_test_split(X_seq, y, test_size=.2)
trainX, testX, trainY, testY = train_test_split(X_seq, y, test_size=.2)


# create the model
######################
model = Sequential()
model.add(Embedding(len(tokenizer.word_index)+1, embedding_dim, input_length=max_length))
model.add(Conv1D(filters=128, kernel_size=6, padding='same', activation='relu'))
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')) 
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(500))
model.add(Dense(top_classes, activation='sigmoid'))
opt = Adam(lr=0.001, decay=1e-6)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())
plot_model(model, to_file='plots/model_plot.png', show_shapes=True, show_layer_names=True)
################

 
save_dir = os.path.join(os.getcwd(), 'models')
model_name = 'keras_cifar10_trained_model.h5'
filepath="model_{epoch:02d}-{val_acc:.2f}.hdf5" 
checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='val_acc',verbose=1, 
                            save_best_only=True,save_weights_only=False)


#learning_rate_reduction = ReduceLROnPlateau(os.path.join(save_dir, filepath), monitor = 'val_acc', patience = 3,
                                            verbose = 1, factor=0.5, min_lr = 0.00001)

											
# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=10):
	scores, histories = list(), list()
	# prepare cross validation
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
	for train_ix, test_ix in kfold.split(dataX):
		# define model
		# select rows for train and test
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		# fit model
		history = model.fit(trainX, trainY, batch_size=128, epochs=25, verbose=1, validation_data=(testX, testY),callbacks=[checkpoint])
		# evaluate model
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))
		# stores scores
		scores.append(acc)
		histories.append(history)
	return scores, histories
 
fnt='Arial'
ticks_font = {'fontname':fnt, 'size':'14'}
legend_font = {'family':fnt, 'size':'12'}
label_font = {'family':fnt, 'size':'14'}
title_font = {'family':fnt, 'size':'14'}
plt.rcParams['figure.figsize'] = [6,4]


# plot diagnostic learning curves

def summarize_diagnostics_1(histories):
    for i in range(len(histories)):
		# plot loss
        plt.xlabel('Epoch')
        plt.ylabel('Loss(%)')		
        plt.rcParams['savefig.dpi'] = 300 
        plt.rcParams['figure.dpi'] = 300              
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')
        plt.savefig('plots/loss.pdf')
        plt.savefig('plots/loss.png',transparent = True)
        plt.savefig('plots/loss.svg',format='svg',transparent = True)		
    plt.show()



def summarize_diagnostics_2(histories):
    for i in range(len(histories)):
		# plot accuracy
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy(%)')		
        plt.rcParams['savefig.dpi'] = 300 
        plt.rcParams['figure.dpi'] = 300              
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['acc'], color='blue', label='train')
        plt.plot(histories[i].history['val_acc'], color='orange', label='test')
        plt.savefig('plots/acc.pdf')
        plt.savefig('plots/acc.png',transparent = True)
        plt.savefig('plots/acc.svg',format='svg',transparent = True)		
    plt.show()



 
# summarize model performance
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	plt.boxplot(scores)
	plt.show()


# run the test harness for evaluating a model
def run_test_harness():
	# evaluate model
	scores, histories = evaluate_model(trainX, trainY)
	# learning curves
	summarize_diagnostics_1(histories)
	summarize_diagnostics_2(histories)	
	# summarize estimated performance
	summarize_performance(scores)

# entry point, run the test harness
run_test_harness()	

model.save("my_model.h5")
model.save_weights('my_model_weights.h5')

train_pred = model.predict(trainX)
test_pred = model.predict(testX)
print("train-acc = " + str(accuracy_score(np.argmax(trainY, axis=1), np.argmax(train_pred, axis=1))))
print("test-acc = " + str(accuracy_score(np.argmax(testY, axis=1), np.argmax(test_pred, axis=1))))
y_test1=np.argmax(testY, axis=1)
test_pred1=np.argmax(test_pred, axis=1)
print(classification_report(y_test1, test_pred1))




##### Compute ROC curve
nb_classes=2

Y_pred = model.predict(testX)
Y_pred = [np.argmax(y) for y in Y_pred] # 
Y_test = [np.argmax(y) for y in testY]

# Binarize the output
Y_test1 = label_binarize(Y_test, classes=[i for i in range(nb_classes)])
Y_pred1 = label_binarize(Y_pred, classes=[i for i in range(nb_classes)])
fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_test1, Y_pred1)
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, color='blue',label='ROC (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(pltout, prop=legend_font, 
           loc='best',frameon=False)
plt.rcParams['savefig.dpi'] = 300 
plt.rcParams['figure.dpi'] = 300 
plt.savefig('plots/ROC.pdf')
plt.savefig('plots/ROC.png',transparent = True)
plt.savefig('plots/ROC.svg',format='svg',transparent = True)	
plt.show()
####


# Zoom in view of the upper left corner.
#plt.figure(2)
#plt.xlim(0, 0.2)
#plt.ylim(0.8, 1)
#plt.plot([0, 1], [0, 1], 'k--')
#plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
#plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
#plt.xlabel('False positive rate')
#plt.ylabel('True positive rate')
#plt.title('ROC curve (zoomed in at top left)')
#plt.legend(loc='best')
#plt.show()


# Compute ROC curve and ROC area for each class
#fpr = dict()
#tpr = dict()
#roc_auc = dict()
#for i in range(nb_classes):
# fpr[i], tpr[i], _ = roc_curve(Y_valid[:, i], Y_pred[:, i])
# roc_auc[i] = auc(fpr[i], tpr[i])


#### save lable
nb_classes=2

Y_pred2 = model.predict(X_seq)
Y_pred2 = [np.argmax(y) for y in Y_pred2]
y_dp = label_binarize(Y_pred2, classes=[i for i in range(nb_classes)])

y1 = lb.fit_transform(subtype)

y2=pd.DataFrame(y1)
y_dp1=pd.DataFrame(Y_pred2)
#heads1=pd.DataFrame(heads)
store = []
store = pd.DataFrame(heads)
store.columns = ['accession']
store['y_hmm'] = y2
store['y_dp'] = y_dp1

store.index=store.index+1
store.to_csv('results_final/lable.csv')
####

#confusion_matrix
from sklearn.metrics import confusion_matrix
from keras.models import load_model
model = load_model('my_model.h5')
Y_pred = model.predict(testX)
Y_pred = [np.argmax(y) for y in Y_pred]
Y_test = [np.argmax(y) for y in testY]
confusionMatrix = confusion_matrix(Y_test, Y_pred)
confusionMatrix
####

#import pickle
 
#with open('trainHistoryDict.txt', 'wb') as file_pi:
    #pickle.dump(history.history, file_pi)

#with open('trainHistoryDict.txt','rb') as file_pi:
#    history=pickle.load(file_pi)

#######



# Imports
#=====================#
import pandas as pd
import numpy as np
from scipy import stats
import random
from Bio import SeqIO
import os
import subprocess

from imblearn.under_sampling import RandomUnderSampler

from Bio.Blast.Applications import NcbiblastpCommandline

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore")

import bioinformatics as bioinf



# Prepare sequences and data
#=====================================================#
# class labels
GH13 = pd.read_csv('results_final/lable.csv')['y_dp']
GH13_AS = GH13[(GH13==1)]
GH13_SH = GH13[(GH13==0)]




# Derive features for machine learning with one-hot encoding
#============================================================#
cat_domain_fasta = 'fasta/GH13_positions_only/GH13_cat.fasta'
sequence_df = bioinf.fasta_to_df(cat_domain_fasta)
X_features = pd.DataFrame() # empty dataframe for storing features

for i in range(len(sequence_df.columns)):
    # Convert amino acids to integers
    X_resid = list(sequence_df.iloc[:,i])
    labelencoder = LabelEncoder()
    X_label = list(labelencoder.fit_transform(X_resid))
    X_resid_unique = sorted(set(X_resid))
    X_label_unique = sorted(set(X_label))
    
    # Map integer labels to amino acids
    label_resid = [X_label.index(num) for num in X_label_unique]
    label_resid = [X_resid[num] for num in label_resid]
    
    # Convert labels to binary features (one-hot encoding)
    onehotencoder = OneHotEncoder()
    X_label = pd.DataFrame(X_label) # convert to 2D array
    X_encoded = onehotencoder.fit_transform(X_label).toarray()
    X_encoded = pd.DataFrame(X_encoded)
    
    # Name encoded features (residue + position, e.g G434)
    X_encoded.columns = ['{0}{1}'.format(res,i+1) for res in label_resid]
    del X_encoded['-{0}'.format(i+1)]  # remove encoded features from gaps
    
    # Append features to dataframe store
    for col in X_encoded.columns:
        X_features[col] = X_encoded[col]    

		
# Randomly split data to validation set and test set
#====================================================#


y=GH13
# Test set data (10% of total data)
SH_test_size = int(0.1 * len(GH13_SH))
AS_test_size = int(0.1 * len(GH13_AS))
SH_test_indices = random.sample(list(GH13_SH.index), SH_test_size)
AS_test_indices = random.sample(list(GH13_AS.index), AS_test_size)
test_indices = SH_test_indices + AS_test_indices
test_indices = sorted(test_indices)

# Validation set data (90% of total data)
val_indices = [x for x in list(y.index) if x not in test_indices]

# X (features) and y for validation and test sets
X_val = X_features.iloc[val_indices,:]
y_val = y.iloc[val_indices]
X_test_sep = X_features.iloc[test_indices,:]
y_test_sep = y.iloc[test_indices]






# Apply random forests to validation set using all features
#=============================================================#

# Empty lists for storing final results
sens_store, spec_store, acc_store, mcc_store, featimp_store = [], [], [], [], []

# Function for evaluating performance
def evalPerf(y_test, y_pred):
    '''Return (sensitivity, specificity, accuracy, MCC, p_value)'''
    cm = confusion_matrix(y_test, y_pred)
    tn, tp, fn, fp = cm[0][0], cm[1][1], cm[1][0], cm[0][1]
    n = tp + fp + tn + fn
    accuracy = (tp + tn)/n * 100
    mcc = ((tp*tn) - (fp*fn))/np.sqrt((tp+fp)*(tn+fn)*(tp+fp)*(tn+fp))
    sens = tp/(tp + fn) * 100 if tp + fp != 0 else 0
    spec = tn/(tn + fp) * 100 if tn + fn != 0 else 0
    table = np.array([[tp, fp], [fn, tn]]) # CBH and EG have same contingency table
    p_value = stats.chi2_contingency(table)[1]
    return [sens, spec, accuracy, mcc, p_value]


# 100 repetitions of 10-fold cross validation
for r in range(100):
    RUS = RandomUnderSampler(random_state=None)
    X_select, y_select = RUS.fit_resample(X_val, y_val)
    X_select, y_select = pd.DataFrame(X_select), pd.Series(y_select)
    
    # 10-fold cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=None)
    kf_indices = kf.split(X_select)
    for train_index, test_index in kf_indices:
        X_train, y_train = X_select.iloc[train_index, :], y_select.iloc[train_index]
        X_test, y_test = X_select.iloc[test_index, :], y_select.iloc[test_index]
        
        # Fit random forest classifier to training data
        classifier = RandomForestClassifier(n_estimators=800, n_jobs=-1)
        classifier.fit(X_train, y_train)
        
        # Test classifier and evaluate performance
        y_pred = classifier.predict(X_test)
        sens, spec, accuracy, mcc, pvalue = evalPerf(y_test, y_pred)
        featimp = list(classifier.feature_importances_)
        
        # Save results
        sens_store.append(sens)
        spec_store.append(spec)
        acc_store.append(accuracy)
        mcc_store.append(mcc)
        featimp_store.append(featimp)

# Average results over all 500 repetitions
store = [np.mean(sens_store), np.std(sens_store), np.mean(spec_store), np.std(spec_store),
         np.mean(acc_store), np.std(acc_store), np.mean(mcc_store), np.std(mcc_store)]
store = pd.DataFrame(store, index=['sens_mean', 'sens_std', 'spec_mean', 'spec_std',
                                   'acc_mean', 'acc_std', 'mcc_mean', 'mcc_std'])
featimp_mean = pd.DataFrame(featimp_store).mean(axis=0)
featimp_std = pd.DataFrame(featimp_store).std(axis=0)
store_featimp = pd.DataFrame([X_val.columns, featimp_mean, featimp_std], 
                             index=['features', 'mean', 'std']).transpose()

# Write results to spreadsheet
store.to_csv('results_final/dp_rf_pred/perf_all.csv')
store_featimp.to_csv('results_final/dp_rf_pred/featimp_all.csv')
     
   
        


        
# Use only top 50 features
#===================================#

# Top 50 features
top50_index = list(store_featimp.sort_values(by='mean', ascending=False).iloc[:50,:].index)
X_val_top50 = X_val.iloc[:,top50_index]

# Empty lists for storing final results
sens_store, spec_store, acc_store, mcc_store, featimp_store = [], [], [], [], []

# 100 repetitions of 10-fold cross validation
for r in range(100):
    RUS = RandomUnderSampler(random_state=None)
    X_select, y_select = RUS.fit_resample(X_val_top50, y_val)
    X_select, y_select = pd.DataFrame(X_select), pd.Series(y_select)
    
    # 10-fold cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=None)
    kf_indices = kf.split(X_select)
    for train_index, test_index in kf_indices:
        X_train, y_train = X_select.iloc[train_index, :], y_select.iloc[train_index]
        X_test, y_test = X_select.iloc[test_index, :], y_select.iloc[test_index]
        
        # Fit random forest classifier to training data
        classifier = RandomForestClassifier(n_estimators=800, n_jobs=-1)
        classifier.fit(X_train, y_train)
        
        # Test classifier and evaluate performance
        y_pred = classifier.predict(X_test)
        sens, spec, accuracy, mcc, pvalue = evalPerf(y_test, y_pred)
        featimp = list(classifier.feature_importances_)
        
        # Save results
        sens_store.append(sens)
        spec_store.append(spec)
        acc_store.append(accuracy)
        mcc_store.append(mcc)
        featimp_store.append(featimp)

# Average results over all 500 repetitions
store_top50 = [np.mean(sens_store), np.std(sens_store), np.mean(spec_store), np.std(spec_store),
         np.mean(acc_store), np.std(acc_store), np.mean(mcc_store), np.std(mcc_store)]
store_top50 = pd.DataFrame(store_top50, index=['sens_mean', 'sens_std', 'spec_mean', 'spec_std',
                                   'acc_mean', 'acc_std', 'mcc_mean', 'mcc_std'])
featimp_mean_top50 = pd.DataFrame(featimp_store).mean(axis=0)
featimp_std_top50 = pd.DataFrame(featimp_store).std(axis=0)
store_featimp_top50 = pd.DataFrame([X_val_top50.columns, featimp_mean_top50, featimp_std_top50], 
                             index=['features', 'mean', 'std']).transpose()

# Write results to spreadsheet
store_top50.to_csv('results_final/dp_rf_pred/perf_top50.csv')
store_featimp_top50.to_csv('results_final/dp_rf_pred/featimp_top50.csv')


# Use only top 20 features
#=================================#

# Top 20 features
top20_index = list(store_featimp.sort_values(by='mean', ascending=False).iloc[:20,:].index)
X_val_top20 = X_val.iloc[:,top20_index]

# Empty lists for storing final results
sens_store, spec_store, acc_store, mcc_store, featimp_store = [], [], [], [], []

# 100 repetitions of 10-fold cross validation
for r in range(100):
    RUS = RandomUnderSampler(random_state=None)
    X_select, y_select = RUS.fit_resample(X_val_top20, y_val)
    X_select, y_select = pd.DataFrame(X_select), pd.Series(y_select)
    
    # 5-fold cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=None)
    kf_indices = kf.split(X_select)
    for train_index, test_index in kf_indices:
        X_train, y_train = X_select.iloc[train_index, :], y_select.iloc[train_index]
        X_test, y_test = X_select.iloc[test_index, :], y_select.iloc[test_index]
        
        # Fit random forest classifier to training data
        classifier = RandomForestClassifier(n_estimators=800, n_jobs=-1)
        classifier.fit(X_train, y_train)
        
        # Test classifier and evaluate performance
        y_pred = classifier.predict(X_test)
        sens, spec, accuracy, mcc, pvalue = evalPerf(y_test, y_pred)
        featimp = list(classifier.feature_importances_)
        
        # Save results
        sens_store.append(sens)
        spec_store.append(spec)
        acc_store.append(accuracy)
        mcc_store.append(mcc)
        featimp_store.append(featimp)

# Average results over all 500 repetitions
store_top20 = [np.mean(sens_store), np.std(sens_store), np.mean(spec_store), np.std(spec_store),
         np.mean(acc_store), np.std(acc_store), np.mean(mcc_store), np.std(mcc_store)]
store_top20 = pd.DataFrame(store_top20, index=['sens_mean', 'sens_std', 'spec_mean', 'spec_std',
                                   'acc_mean', 'acc_std', 'mcc_mean', 'mcc_std'])
featimp_data = pd.DataFrame(featimp_store, columns=X_val_top20.columns)
featimp_mean_top20 = pd.DataFrame(featimp_store).mean(axis=0)
featimp_std_top20 = pd.DataFrame(featimp_store).std(axis=0)
store_featimp_top20 = pd.DataFrame([X_val_top20.columns, featimp_mean_top20, featimp_std_top20], 
                             index=['features', 'mean', 'std']).transpose()

# Write results to excel spreadsheet
featimp_data.to_csv('results_final/dp_rf_pred/featimp_top20_fulldata.csv')
store_top20.to_csv('results_final/dp_rf_pred/perf_top20.csv')
store_featimp_top20.to_csv('results_final/dp_rf_pred/featimp_top20.csv')
        



# Train top20 classifier and test on test set
#===============================================#
RUS = RandomUnderSampler(random_state=None)
X_select, y_select = RUS.fit_resample(X_val_top20, y_val)
classifier = RandomForestClassifier(n_estimators=800, n_jobs=-1)
classifier.fit(X_select, y_select)
y_pred = classifier.predict(X_test_sep.iloc[:,top20_index])
cm = confusion_matrix(y_test_sep, y_pred)
tn, tp, fn, fp = cm[0][0], cm[1][1], cm[1][0], cm[0][1]
sens, spec, accuracy, mcc, pvalue = evalPerf(y_test_sep, y_pred)
store = pd.DataFrame([tp, fp, tn, fn, sens, spec, accuracy, mcc], 
                     index=['tp', 'fp', 'tn', 'fn', 'sens', 'spec', 'acc', 'mcc'])
store.to_csv('results_final/dp_rf_pred/perf_test_set.csv')

        




# Position specific rules from top 50 features (Supporting Information)
#===============================================================================#
store = [] # empty list for storing results
for col in X_val_top50.columns:
    # Test the rule, feature => SH
    y_pred = X_val_top50[col]
    cm = confusion_matrix(y_val, y_pred)
    perf = evalPerf(y_val, y_pred)
    
    # Test the rule, not feature => SH
    y_pred1 = [1 if x==0 else 0 for x in y_pred]
    cm1 = confusion_matrix(y_val, y_pred1)
    perf1 = evalPerf(y_val, y_pred1)
    
    # Add rule (i.e. the rule with the highest MCC)
    if perf[3] > perf1[3]:
        perf = ['{0}=>SH'.format(col)] + perf
        store.append(perf)
    else:
        perf1 = ['not {0}=>SH'.format(col)] + perf1
        store.append(perf1)

store = pd.DataFrame(store, columns=['rule', 'sensitivity', 
                                     'specificity', 'accuracy', 'mcc', 'pvalue'])
store.to_csv('results_final/dp_rf_pred/position_rules.csv')
