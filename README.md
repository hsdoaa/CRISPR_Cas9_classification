# CRISPR_Cas9_classification

These are a group of classifiers that classify the average fold change in Crisper-cas9 dataset into four type of classes, where each class represent a quartile of the average fold change range. These models include:

Support vector machine (SVR)

k-nearest neighbors (KNN)

Random forest (RF) 

Decision tree (DT)

Logistic regression (LR)

Naive bayes (NB)

Convolutional neural network (CNN)



All the classifers are implemented in python using Scikit-learn library except  CNN is implemented in Keras with tensorflow back end. All classifiers except CNN are implemented in clf.py, clf_validate.py, clf_validate_onehot.py, where each script uses a different set of features exracted from the crisper-cas9 dataset for training the classifer. CNN is implemented in cnn_validate.py, cnn_embedding.py, cnn_validate_onehot.py, where each CNN script uses a different set of features exracted from the crisper-cas9 dataset for training the CNN. 

