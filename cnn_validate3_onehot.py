#https://www.dataquest.io/blog/learning-curves-machine-learning/

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
   lb = LabelBinarizer()
   lb.fit(y_test)
   y_test = lb.transform(y_test)
   y_pred = lb.transform(y_pred)
   return roc_auc_score(y_test, y_pred, average=average)

df = pd.read_csv("train_dataset.csv")
df = df.dropna() # To drop Null values 
print(df.shape)

df = df.drop_duplicates('sgRNA_sequence', keep='last')
print("&&&&&&&&")

print(df.shape)

print("&&&&&&&&")

print(df.head())
#########################################################
target_value_list=list(df.iloc[:, 12]) 
print("22222222222", len(target_value_list))
print(target_value_list[0])
target_value_class=[]
for i in target_value_list:
    if type(i)!=str:
        if i>=0 and i<0.052:
            i=0#'Min_Q1'
            target_value_class.append(i) 
        elif i>=0.052 and i<0.88:
            i=1#'Q1_Q2'
            target_value_class.append(i) 
        elif i>=0.88 and i<1.63:
            i=2#'Q2_Q3'
            target_value_class.append(i)
        else:
            i=3#'Q3_Max'
            target_value_class.append(i)

print("length of target_value_class",len(target_value_class))
print(target_value_class[0:50])
df['label']=target_value_class
print(df.head())

######################################################################

print("#######################")
df2 = pd.read_csv("test_dataset.csv")
df2 = df2.dropna() # To drop Null values
print(df2.shape) 
df2 = df2.drop_duplicates('sgRNA_sequence', keep='last')
print("&&&&&&&&")
print(df2.shape)
print("&&&&&&&&")
print(df2.head())
#########################################################


target_value_list=list(df2.iloc[:, 12]) 
print("22222222222", len(target_value_list))
print(target_value_list[0])
target_value_class=[]
for i in target_value_list:
    if type(i)!=str:
        if i>=0 and i<0.031:
            i=0#'Min_Q1'
            target_value_class.append(i) 
        elif i>=0.031 and i<0.84:
            i=1#'Q1_Q2'
            target_value_class.append(i) 
        elif i>=0.84 and i<1.59:
            i=2#'Q2_Q3'
            target_value_class.append(i)
        else:
            i=3#'Q3_Max'
            target_value_class.append(i)

print("length of target_value_class",len(target_value_class))
print(target_value_class[0:50])
df2['label']=target_value_class
print(df2.head())


##################################################################
#combine onehot_encoding of train and test
union_reference_kmer_set=set(df.iloc[:, 1]).union(set(df2.iloc[:, 1]))
union=list(union_reference_kmer_set)
print(len(union))
df['sgRNA_sequence']=pd.Categorical(df['sgRNA_sequence'], categories=list(union))
df2['sgRNA_sequence']=pd.Categorical(df2['sgRNA_sequence'], categories=list(union))
####################################################################

####################################################################

features_train=['Efficiency','Specificity','BS_Length','PEEK','GWAS','Distance_exon_BS (D)','sgRNA_Conc_Q0','sgRNA_Conc_T5_K562_Rep1_normalized']

#X_train = df[features_train]
#print(X_train.shape)
#X_train= scaler.fit_transform(X_train)

X_train = df[features_train]
#insert onehot encoding of reference-kmer
Onehot=pd.get_dummies(df['sgRNA_sequence'], prefix='sgRNA_sequence')
X_train= pd.concat([X_train,Onehot],axis=1)
a,b=X_train.shape
print("b=",b)

features_test=['Efficiency','Specificity','BS_Length','PEEK','GWAS','Distance_exon_BS (D)','sgRNA_Conc_Q0','sgRNA_Conc_T5_K562_Rep2_normalized']
#X_test = df2[features_test]
#print(X_test.shape)
#X_test= scaler.fit_transform(X_test)

X_test = df2[features_test]
#insert onehot encoding of reference-kmer
Onehot=pd.get_dummies(df2['sgRNA_sequence'], prefix='sgRNA_sequence')
X_test= pd.concat([X_test,Onehot],axis=1)
c,d=X_test.shape
print("d=",d)

X_train= scaler.fit_transform(X_train)
X_test= scaler.fit_transform(X_test)

y_train = df['label']
print(y_train.head())
y_test = df2['label']
print(y_test.head())

#######################################################  
X_train = np.expand_dims(np.random.normal(size=(a, b)),axis=-1)  #shap input-train to have 3 D tensor in order to be processed with conv1D
# Ytrain => [213412, 10]
#y_train = np.random.choice([0,1,2,3], size=(a,1))y_train = to_categorical(y_train)

X_test = np.expand_dims(np.random.normal(size=(c, d)),axis=-1)    #shap input-test to have 3 D tensor in order to be processed with conv1D
# Ytrain => [213412, 10]
#y_test = np.random.choice([0,1,2,3], size=(c,1))


####################################################################
#Sequence/signal classification with 1D convolutions: https://keras.io/getting-started/sequential-model-guide/  ,  https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Flatten,BatchNormalization,LeakyReLU
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D,GlobalMaxPooling1D
from tensorflow.keras.optimizers import SGD, Adam,RMSprop
from tensorflow.keras.utils import to_categorical
#one_hot_label = to_cateorical(input_labels)
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping,ModelCheckpoint,ReduceLROnPlateau, Callback
import matplotlib.pyplot as plt
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report #for classifier evaluation
from sklearn.metrics import roc_auc_score # for printing AUC

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#Implement cnn model4 for training embedding layer, available at https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/


learning_rate=0.0001

#################
model = Sequential()
model.add(Conv1D(filters=1024, kernel_size=3, activation='relu',input_shape=(b, 1))) #or kernel_size=8 and MaxPooling1D(pool_size=1)
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=1024, kernel_size=3, activation='relu'))
model.add(Flatten())
#model.add(Dense(10, activation='relu'))
model.add(Dense(4, activation='softmax'))
#########


print(model.summary())


callbacks_list = [ModelCheckpoint(filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',monitor='val_loss', save_best_only=True),EarlyStopping(monitor='acc', patience=1)]

model.compile(loss='categorical_crossentropy',
                optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

BATCH_SIZE = 256
EPOCHS = 5

history = model.fit(X_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.1,
                      verbose=1)

# evaluate the cnn model on the same dataset
_, accuracy = model.evaluate(X_test, y_test, verbose=0)
#_, accuracy = model.evaluate(X_test, y_test, verbose=1)
print('Accuracy: %.2f' % (accuracy*100))
loss_train, accuracy_train = model.evaluate(X_train, y_train, verbose=1)
loss_test, accuracy_test = model.evaluate(X_test, y_test, verbose=1)
print('Accuracy on training: %.2f' % (accuracy_train*100))
print('Accuracy on testing: %.2f' % (accuracy_test*100))
#print('Test loss:', loss_test)



train_loss = history.history['loss']
val_loss   = history.history['val_loss']
train_acc  = history.history['accuracy']
val_acc    = history.history['val_accuracy']
xc         = range(50)

#plot epochs versus accuracy curve.

plt.plot(train_acc, label="Training accuracy")
plt.plot(val_acc, label="Validation accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc="best")
plt.show()

plt.plot(train_loss, label="Training loss")
plt.plot(val_loss, label="Validation loss")
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend(loc="best")
plt.show()

# make class predictions with the model
#y_pred = model.predict_classes(X_test) 

y_pred =np.argmax(model.predict(X_test), axis=-1)

import numpy as np   #to handle the error Classification metrics can't handle a mix of multilabel-indicator and multiclass targets
y_test=np.argmax(y_test, axis=1)
y_test[1]
print(y_test[1])

print(classification_report(y_test, y_pred))

print(y_test.shape)
print(y_pred.shape)
y_test=y_test.reshape(-1,1)  #reshape from 1 column array to two column array, because some np functions does not like missing dimensions (-,)
y_pred=y_pred.reshape(-1,1)
print(y_test.shape)
print(y_pred.shape)

print("AUC=",multiclass_roc_auc_score(y_test,y_pred)) #https://medium.com/@plog397/auc-roc-curve-scoring-function-for-multi-class-classification-9822871a6659











