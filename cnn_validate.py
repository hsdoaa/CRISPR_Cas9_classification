

#https://www.dataquest.io/blog/learning-curves-machine-learning/

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

print("&&&&&&&&")

print(df.head())
print(df.shape)

print("###",df['K_145-F3'])

#remove data points in the first quartile (i.e.,if they are >=0 and <0.052):
#df_filtered=df.query('K_145-F3>=0 & K_145-F3<0.052')

df_filtered = df[df['K_145-F3']>=0.052] 

print("======")

print(df_filtered.shape)
print(df_filtered.describe())


#shuffle the test and train datasets
from sklearn.utils import shuffle
#df= shuffle(df)

 
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
#df_3classes = df[df['label']!='Min_Q1'] #to remove datapoints of 'Min_Q1' class
#print("$$$$",df_3classes.shape)


######################################################################

print("#######################")
df2 = pd.read_csv("test_dataset.csv")
df2 = df2.dropna() # To drop Null values 

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
#df2_3classes = df2[df2['label']!='Min_Q1'] #to remove datapoints of 'Min_Q1' class
#print("@@@@@",df2_3classes.shape)


####################################################################

#shuffle the test and train datasets
from sklearn.utils import shuffle
df= shuffle(df)
df2=shuffle(df2)

features_train=['Efficiency','Specificity','BS_Length','PEEK','GWAS','Distance_exon_BS (D)','sgRNA_Conc_Q0','sgRNA_Conc_T5_K562_Rep1_normalized']

X_train = df[features_train]
print(X_train.shape)
print("ppp",X_train.info()) 
X_train= scaler.fit_transform(X_train)
a,b=X_train.shape

print("b=",b)
features_test=['Efficiency','Specificity','BS_Length','PEEK','GWAS','Distance_exon_BS (D)','sgRNA_Conc_Q0','sgRNA_Conc_T5_K562_Rep2_normalized']
X_test = df2[features_test]
print(X_test.shape)
X_test= scaler.fit_transform(X_test)
c,d=X_test.shape

print("d=",d)
y_train = df['label']
print("0000",y_train.head())
y_test = df2['label']
print("%%%",y_test.head())

##########################################################
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




###########################

v=b+d
learning_rate=0.0001
#metrics=['accuracy']
callbacks = []

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
	#self.val_losses=[]

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
	#self.val_losses.append(logs.get('val_loss'))


#implement cnn model2
#implement cnn model # modify the model at https://androidkt.com/keras-confusion-matrix-in-tensorboard/
model = Sequential()
model.add(Conv1D(32, 3, activation='relu', input_shape=(b,1)))  #relu can be changed to elu and selu ELU, and LeakyReLU(alpha=0.01).
model.add(MaxPooling1D(2))
model.add(Conv1D(64, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Flatten())
model.add(Dense(4, activation='softmax'))


callbacks_list = [ModelCheckpoint(filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',monitor='val_loss', save_best_only=True),EarlyStopping(monitor='acc', patience=1)]

model.compile(loss='categorical_crossentropy', # 'binary_crossentropy'
                optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

BATCH_SIZE = 400
EPOCHS = 50

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
y_pred =np.argmax(model.predict(x), axis=-1)
import numpy as np
y_test=np.argmax(y_test, axis=1)
y_test[1]
print(y_test[1])
print(classification_report(y_test, y_pred))
print(type(y_test))
print(np.unique(y_test)) #for pandas series y_test.unique()
print("y_test_dim",y_test.shape)
print(type(y_pred))
print(y_pred)
print("y_pred_dim",y_pred.shape)
y_test=y_test.reshape(-1,1)  #reshape from 1 column array to two column array, because some np functions does not like missing dimensions (-,)
y_pred=y_pred.reshape(-1,1)
print(y_test.shape)
print(y_pred.shape)
print("AUC=",multiclass_roc_auc_score(y_test,y_pred)) #https://medium.com/@plog397/auc-roc-curve-scoring-function-for-multi-class-classification-9822871a6659
#auc=roc_auc_score(y_test,y_pred,multi_class='ovr') #,multi_class='ovr'
#auc = float("{0:.3f}".format(auc))
#print("AUC=",auc)
####################################
