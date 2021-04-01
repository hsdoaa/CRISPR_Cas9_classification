from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping,ModelCheckpoint,ReduceLROnPlateau, Callback

'''
tok = Tokenizer()
#tok.fit_on_texts(["this comment is not toxic"]) 
tok.fit_on_texts(["AACAGGGGCAGTGAACAAGA"])
print(tok.texts_to_sequences(["AACAGGGGCAGTGAACAAGA"])) 
print(tok.texts_to_sequences(["AACAGGGGCAGTGAACAAGA"]))
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size )
'''

##################################################

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer

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
print(df.info())
print(df.head())
print(df.shape)



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

#remove data points in the first quartile (i.e.,if they are >=0 and <0.031):

df2_filtered = df2[df2['K_146-F3']>=0.031]
print("======") 
print(df2_filtered.shape)
print(df2_filtered.describe())


#df2=shuffle(df2)
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


print(df2.info())
print(df2.shape)


####################################################################

#shuffle the test and train datasets
from sklearn.utils import shuffle
#df= shuffle(df)
#df2=shuffle(df2)

tokenizer = Tokenizer()


tokenizer.fit_on_texts(df.sgRNA_sequence.values)
post_seq_train = tokenizer.texts_to_sequences(df.sgRNA_sequence.values)
import itertools
post_seq_train= list(itertools.chain(*post_seq_train)) # to flat list of lists to one list

print(post_seq_train)
print(type(post_seq_train))
print("^^^^",len(post_seq_train))
tokenizer.fit_on_texts(df2.sgRNA_sequence.values)
post_seq_test = tokenizer.texts_to_sequences(df2.sgRNA_sequence.values)
print("VVVV",len(post_seq_test))
post_seq_test= list(itertools.chain(*post_seq_test)) # to flat list of lists to one list
#df_seq_train=pd.DataFrame(post_seq_train)
#df_seq_test=pd.DataFrame(post_seq_test)


#print(df_seq_train.shape)
#print(df_seq_test.shape)

print("======================")

features_train=['Efficiency','Specificity','BS_Length','PEEK','GWAS','Distance_exon_BS (D)','sgRNA_Conc_Q0','sgRNA_Conc_T5_K562_Rep1_normalized']

X_train = df[features_train]
print("shape of X_train=",X_train.shape)
X_train['seq_train']=post_seq_train
#X_train= pd.concat([X_train,df_seq_train],axis=1)
print("shape of X_train=",X_train.shape)

X_train= scaler.fit_transform(X_train)
a,b=X_train.shape

print("b=",b)
features_test=['Efficiency','Specificity','BS_Length','PEEK','GWAS','Distance_exon_BS (D)','sgRNA_Conc_Q0','sgRNA_Conc_T5_K562_Rep2_normalized']
X_test = df2[features_test]
print("shape of X_test=",X_test.shape)
X_test['seq_test']=post_seq_test
#X_test= pd.concat([X_test,df_seq_test],axis=1)
print("shape of X_test=",X_test.shape)
X_test= scaler.fit_transform(X_test)
c,d=X_test.shape

print("d=",d)
y_train = df['label']
y_train=to_categorical(y_train)
#print("0000",y_train.head())
y_test = df2['label']
y_test=to_categorical(y_test)
#print("%%%",y_test.head())

vocab_size = len(tokenizer.word_index) + 1
print("vocab_size=",vocab_size)
###################################################

inputs = Input(shape=(b, ))
embedding_layer = Embedding(vocab_size,
                            128,
                            input_length=b)(inputs)
x = Flatten()(embedding_layer)
x = Dense(32, activation='relu')(x)

predictions = Dense(4, activation='softmax')(x)
model = Model(inputs=[inputs], outputs=predictions)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
filepath="weights-simple.hdf5"
callbacks_list = [ModelCheckpoint(filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',monitor='val_loss', save_best_only=True),EarlyStopping(monitor='acc', patience=1)]

checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#history = model.fit([X_train], batch_size=64, y_train=to_categorical(y_train), verbose=1, validation_split=0.25, 
#          shuffle=True, epochs=5, callbacks=[checkpointer])

history = model.fit(X_train, y_train,  batch_size=64, verbose=1, validation_split=0.25, 
         shuffle=True, epochs=5, callbacks=[checkpointer])




import matplotlib.pyplot as plt
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report #for classifier evaluation
from sklearn.metrics import roc_auc_score # for printing AUC


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
y_pred=np.argmax(X_test,axis=1)

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











