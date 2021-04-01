#https://www.dataquest.io/blog/learning-curves-machine-learning/

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler

#scaler = MinMaxScaler()
min_max_scaler = MinMaxScaler()

df = pd.read_csv("train_dataset.csv")
df = df.dropna() # To drop Null values 

print(df.shape)

print("&&&&&&&&")

print(df.head())

print(df['Distance_exon_BS (D)'])

df['Distance_exon_BS (D)']=abs(df['Distance_exon_BS (D)'])

print("&&&&&&&&")
print(df['Distance_exon_BS (D)'])

#########################################################
target_value_list=list(df.iloc[:, 10]) 
print("22222222222", len(target_value_list))
print(target_value_list[0])
target_value_class=[]
for i in target_value_list:
    if type(i)!=str:
        if i>=0 and i<0.11:
            i='Min_Q1'
            target_value_class.append(i) 
        elif i>=0.11 and i<0.81:
            i='Q1_Q2'
            target_value_class.append(i) 
        elif i>=0.81 and i<1.375:
            i='Q2_Q3'
            target_value_class.append(i)
        else:
            i='Q3_Max'
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

print("&&&&&&&&")

print(df2.head())
#########################################################
target_value_list=list(df2.iloc[:, 10])
print("22222222222", len(target_value_list))
print(target_value_list[0])
target_value_class=[]
for i in target_value_list:
    if type(i)!=str:
        if i>=0 and i<0.1:
            i='Min_Q1'
            target_value_class.append(i) 
        elif i>=0.1 and i<0.81:
            i='Q1_Q2'
            target_value_class.append(i) 
        elif i>=0.81 and i<1.43:
            i='Q2_Q3'
            target_value_class.append(i)
        else:
            i='Q3_Max'
            target_value_class.append(i)

print("length of target_value_class",len(target_value_class))
print(target_value_class[0:50])
df2['label']=target_value_class
print(df2.head())






####################################################################
columns=['Efficiency','Specificity','BS_Length','PEEK','GWAS','Distance_exon_BS (D)','sgRNA_Conc_Q0','sgRNA_Conc_T5_K562_Rep1_normalized', 'label']
features=['Efficiency','Specificity','BS_Length','PEEK','GWAS','Distance_exon_BS (D)','sgRNA_Conc_Q0','sgRNA_Conc_T5_K562_Rep1_normalized']
#features=['BS_Length','PEEK','GWAS']
#features_1=['Efficiency']
#features_2=['Specificity']
#features_3=['BS_Length']
#features_4=['Distance_exon_BS (D)']
#features_5=['sgRNA_Conc_Q0']
#features_6=['sgRNA_Conc_T5_K562_Rep1_normalized']

dataset_train= df[columns]
print(dataset_train.info())
#dataset_train.to_csv('train.csv', index=False)

#######################################################################################################


columns2=['Efficiency','Specificity','BS_Length','PEEK','GWAS','Distance_exon_BS (D)','sgRNA_Conc_Q0','sgRNA_Conc_T5_K562_Rep2_normalized', 'label']
features2=['Efficiency','Specificity','BS_Length','PEEK','GWAS','Distance_exon_BS (D)','sgRNA_Conc_Q0','sgRNA_Conc_T5_K562_Rep2_normalized']
#features2=['BS_Length','PEEK','GWAS']
dataset_test= df2[columns2]
print(dataset_test.info())
#dataset_test.to_csv('test.csv', index=False)
###################################################
#Feature Engineering
#=========================
#convert label from object to category type
#=================================
print("bbb",dataset_train.Efficiency)
print("ccc",dataset_test.Efficiency)
dataset_train['label'] = dataset_train.label.astype('category')
dataset_test['label'] = dataset_test.label.astype('category')

#descretize numerical features
##############################

#dataset_train['Efficiency'] = pd.qcut(df['Efficiency'], q=4)

#print("MMM",dataset_train['Efficiency'])

#print(dataset_train.info())
#print(dataset_test.info())
########################################################

X_train = dataset_train[features]
print(X_train.shape)
y_train = dataset_train['label']
print(y_train.head())

X_test = dataset_test[features2]
print(X_test.shape)
y_test = dataset_test['label']
print(y_test.head())

#feature scaling
X_train= min_max_scaler.fit_transform(X_train)
X_test=min_max_scaler.fit_transform(X_test)
#X_train= scaler.fit_transform(X_train)
#X_test= scaler.fit_transform(X_test)
#######################################################  
  
# training a DescisionTreeClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
##########
#clf=LogisticRegression(C=1e4)
#clf=LogisticRegression(random_state=random.seed(1234))
#clf=DecisionTreeClassifier(max_depth = 2)
#clf=KNeighborsClassifier(n_neighbors=2)#(n_neighbors=3)default#n_neighbors=5
clf=svm.SVC()
#clf=RandomForestClassifier(n_estimators=30, max_depth=10, random_state = random.seed(1234))#random_state=0)
#clf = BaggingClassifier(n_estimators=500,max_samples=0.1,base_estimator=LogisticRegression(random_state=random.seed(1234)))
#clf = BaggingClassifier()
#clf = GaussianNB()
print(clf)
model = clf.fit(X_train, y_train) 
#model = clf.fit(X_train, y_train.ravel()) 
y_pred = model.predict(X_test)
###############################################
#y_prob = clf.predict_proba(X_test.values.reshape(-1, 1))
#y_prob = y_prob[:,1]
# Evaluate the model: Model Accuracy, how often is the classifier correct
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report #for classifier evaluation
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score # for printing AUC
from sklearn.metrics import confusion_matrix
# creating a confusion matrix 
cm = confusion_matrix(y_test, y_pred )
print(cm)



print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)
 
print(classification_report(y_test, y_pred))
#auc=roc_auc_score(y_test.round(),y_pred,multi_class="ovr",average=None)
#auc = float("{0:.3f}".format(auc))
#print("AUC=",auc)

#true negatives c00, false negatives C10, true positives C11, and false positives C01 
#tn c00, fpC01, fnC10, tpC11 
print('CF=',confusion_matrix(y_test, y_pred))
l=confusion_matrix(y_test, y_pred)#https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9
print('TN',l.item((0, 0)))
print('FP',l.item((0, 1)))
print('FN',l.item((1, 0)))
print('TP',l.item((1, 1)))
#print(type(X_train), type(y_train))



print("$$$$$",features)
print("@@@@@",features2)
'''
#plot learning curve: works with all classifier and all features except x(padded signal) as it leads to error with SVM 
#References:https://medium.com/@datalesdatales/why-you-should-be-plotting-learning-curves-in-your-next-machine-learning-project-221bae60c53
import matplotlib.pyplot as plt
plc. plot_learning_curves(clf, X_train, y_train, X_test, y_test)

# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()


#plot ROC curve: https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)


# Print ROC curve
plt.plot(fpr,tpr)
plt.title("ROC Curve")
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show() 

###############################################
'''
  

