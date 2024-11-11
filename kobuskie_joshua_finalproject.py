# %% [markdown]
# # NJIT Fall 2024 CS634: Data Mining Final Project #
# ## Joshua Kobuskie ##
# ### November 17, 2024 ###

# %% [markdown]
# ##### Supervised Data Mining (Classification) Binary Classification Only #####

# %% [markdown]
# *Implement 3 different classification algorithms in Python. One of them is Random Forest, the second one is from the deep learning list in the “Appendix → Additional Option: Deep Learning”, and the third is from the list of algorithms in “Appendix → Additional Option: Algorithms” on 1 dataset of your choice (each of the three algorithms must run on the same dataset).*

# %% [markdown]
# *NOTE: This is not from scratch implementation, just use the existing libraries to implement the algorithms, but the performance metrics must be calculated manually. You may use “confusion_matrix” library to get TP, TN, FP, FN ONLY, then calculate the FPR, FNR, etc.… using the formulas from the slides.*

# %% [markdown]
# *Sources of data are listed in the Appendix “Additional Option: Sources of Data” or use your own.*

# %% [markdown]
# *Your final term project documentation must clearly indicate the algorithms and dataset you used in the project.*  
# 
# The classification algorithms that I have chosen to implement and evaluate in this project are the Random Forest Classifier, the Convolutional 1D network, and the naive Bayes algorithm. 
# The dataset I have chosen to use in this project is the KDDCup99 dataset from the Scikit-learn real world datasets. This dataset contains TCP network connection data with about 5 million training records and 2 million test records. Each record is labeled as normal, or as a specific attack type. To ensure that this data will meet the criteria for binary classification used in this project, the data will be separated into 2 classes of either normal or attack, rather than the specific classes of attacks detailed within the dataset. This also models the intention of the dataset by prioritizing attack detection rather than attack type identification. This is evident as the training data contains 24 types of attack, but the test data contains an additional 14 types of attacks not previously seen in the training data. By attempting to identify attacks rather than attack types, this will test the models' ability to detect known, and unknown attacks based on their understanding of normal network data. The models I will implement will be able to classify records as either normal TCP network connection data or abnormal TCP network connection data indicative of an attack.

# %% [markdown]
# *In addition to the general submission rules and grading, include the websites where the software and complete dataset can be downloaded.*  
# 
# The Scikit-learn software can be downloaded at the following link: https://scikit-learn.org/stable/install.html  
# The Keras software can be downloaded at the following link: https://keras.io  
# The KDDCup99 dataset can be downloaded in its entirety at the following link: https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

# %% [markdown]
# The KDDCup99 dataset is first imported from Scikit-learn and inspected.

# %%
from sklearn.datasets import fetch_kddcup99
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
dataset = fetch_kddcup99(as_frame=True).frame
# print(dataset["labels"].value_counts())
# print(dataset.info())

# %% [markdown]
# All values in the dataset are stored as objects and will be converted into integers, floats, or strings based on the appropriate data type for easier processing.

# %%
for col in dataset.columns:
    try:
        dataset[col] = pd.to_numeric(dataset[col])
    except:
        dataset[col] = dataset[col].apply(lambda x: x.decode("utf-8"))
        dataset[col] = dataset[col].astype("string")

# print(dataset.info())

# %% [markdown]
# While the entire dataset provides valuable information to improve the accuracy of the models, it is too large and slows down training significantly. To improve training times, the dataset will be reduced to 10% of the size while preserving the class proportions through stratification based on the labels.

# %%
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    dataset = dataset.groupby("labels", group_keys=False).apply(lambda x: x.sample(frac=0.1, random_state=42)).reset_index(drop=True)
# print(dataset.info())

# %% [markdown]
# This dataset contains all the data needed for classification, but currently stores the protocol_type, service, flag, and labels as strings. These values will be one-hot encoded rather than being preserved as categorical variables to prevent ordinal implications in the models. This data also presents a multi-classification problem rather than a binary classification. To transform this into a binary classification problem, I will use two class labels of normal and attack and convert the current dataset labels accordingly. All non-normal labels will be considered an attack for training and testing purposes.

# %%
def transformLabel(col):
    if col == "normal.":
        return 0
    else:
        return 1

#1-hot encode
dataset = pd.get_dummies(dataset, columns = ["protocol_type", "service", "flag"])

#Normal will be 0, attack will be 1
labelDict = {"normal":0, 0: "normal", "attack":1, 1:"attack"}
dataset["labels"] = dataset["labels"].apply(transformLabel)

#Sort data for stratified results later
dataset = dataset.sort_values(by=["labels"], ignore_index=True)

#Save index of starting 1
attackIndex = dataset.index[dataset.labels == 1][0]

# print(dataset.info())

# %% [markdown]
# Now that the data has been transformed, I prepare this dataset for training by separating the X dimensions and the Y labels.

# %%
xdf = dataset.loc[:, dataset.columns != "labels"]
ydf = dataset.loc[:, "labels"]
# print(xdf.info())
# print(ydf.info())

# %% [markdown]
# In the KDDCup99 dataset, there is an imbalance in the number of normal connections and attack connections with a bias toward attack connection data. This bias is visualized below.

# %%
# plt.bar(["Attack", "Normal"], ydf.value_counts(), color=["red", "blue"])
# plt.xlabel("Network Connection Class")
# plt.ylabel("Number of Entries")
# plt.title("Kddcup99 Data Classes")
# plt.show()
print("#################### Checking for Data Imbalance ####################")
print("Attack Entries: {}, Normal Entries: {}, Total Entries: {}".format(ydf.value_counts()[1], ydf.value_counts()[0], ydf.count()))
print("Attack Entries: {:.2f}%, Normal Entries: {:.2f}%".format(ydf.value_counts()[1] / ydf.count() * 100, ydf.value_counts()[0] / ydf.count() * 100))
print()

# %% [markdown]
# To overcome this imbalance in data, either the normal data could be over-sampled or the attack data could be under-sampled. While these options would balance the dataset, the data mining techniques used in this project are able to effectively work with small imbalances in data and neither of these approaches are necessary to achieve high model performance. In the real world, it is likely that an imbalance in the data would exist as well, and thus the small imbalance has been disregarded. To ensure that the classes are proportionally utilized during training and testing, a stratified sampling approach will be used during the k-fold cross validation.

# %% [markdown]
# To visualize the correlation of the variables, a correlation matrix is used. This will help to identify data for dimensional reduction.

# %%
#create and display correlation matrix
corrMat = xdf.corr().abs()

# fig, axis = plt.subplots(figsize=(30, 30))
# sns.heatmap(corrMat, annot=True, linewidths=.5, fmt='.2f', ax=axis)
# plt.title("Correlation Matrix") 
# plt.xlabel("Features") 
# plt.ylabel("Features")
# plt.show()

# %% [markdown]
# Variables where the value does not change, resulting in white lines within the correlation matrix, are dropped because they provide no value to the models. Variables with highly correlated values are also dropped to prevent multicollinearity. The new correlation matrix is created with the reduced dimensionality dataset.

# %%
#Since the correlation matrix is symetrical, only look at one half
upperTriangle = corrMat.where(np.triu(np.ones(corrMat.shape), k=1).astype(bool))

#Drop features with high correlation above 0.8
xdf = xdf.drop([col for col in upperTriangle.columns if any(upperTriangle[col] > 0.8)], axis=1)

#Drop columns with only one unique value
xdf = xdf.drop(columns=[col for col in xdf.columns if xdf[col].nunique() == 1])

#Create and display correlation matrix
corrMat = xdf.corr().abs()

# fig, axis = plt.subplots(figsize=(30, 30))
# sns.heatmap(corrMat, annot=True, linewidths=.5, fmt='.2f', ax=axis)
# plt.title("Trimmed Dataset Correlation Matrix") 
# plt.xlabel("Features") 
# plt.ylabel("Features")
# plt.show()

# %% [markdown]
# *The performance metrics must be calculated manually. You may use “confusion_matrix” library to get TP, TN, FP, FN ONLY, then calculate the FPR, FNR, etc.… using the formulas from the slides.*  
# 
# The following performance metrics are calculated by the getMetrics function: True Positive, True Negative, False Positive, False Negative, Positive, Negative, True Positive Rate, True Negative Rate, False Positive Rate, False Negative Rate, Recall/ Sensitivity, Precision, F1 Score, Accuracy, Error Rate, Balanced Accuracy, True Skill Statistics, Heidke Skill Score, Specificity, Negative Predictive Value, False Discovery Rate, Brier Score, Brier Skill Score, and the Area Under ROC Curve. Each performance metric is manually calculated using only the True Positive, True Negative, False Positive, and False Negative values from the confusion matrix, except for the Brier Score and the AUC which utilize a library for calculation. The getMetrics function will be called to evaluate the performance of each model after training in the k-fold cross validation.
# 
# **The professor has made the exception that both the Brier Score and the AUC can be calculated using a library.**

# %%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import brier_score_loss
from sklearn.metrics import roc_auc_score

#Add val to all to make variables instead of constants
def getMetrics(yTest, yPred, yProb):
    
    #Use Confusion Matrix to get TP, TN, FP, FN ONLY
    confMat = confusion_matrix(yTest, yPred, labels=[0,1])
    TPval, TNval, FPval, FNval = confMat[1][1], confMat[0][0], confMat[0][1], confMat[1][0]

    #Calculate performance metrics based on class slides
    Pval = TPval + FNval
    Nval = TNval + FPval

    TPRval = TPval / Pval
    TNRval = TNval / Nval
    FPRval = FPval / Nval
    FNRval = FNval / Pval

    r = TPval / Pval
    p = TPval / (TPval + FPval)

    #F1 can encounter 0 division error
    if p + r == 0:
        f1 = 0.0
    else:
        f1 = 2 * (p * r) / (p + r)

    Acc = (TPval + TNval) / (Pval + Nval)
    Err = (FPval + FNval) / (Pval + Nval)

    BACCval = (TPRval + TNRval) / 2
    TSSval = (TPval / (TPval + FNval)) - (FPval / (FPval+TNval))
    HSSval = (2 * (TPval * TNval - FPval * FNval)) / ((TPval + FNval) * (FNval + TNval) + (TPval + FPval) * (FPval + TNval))

    SPCval = TNval / (FPval + TNval)
    NPVval = TNval / (TNval + FNval)
    FDRval = FPval / (FPval + TPval)

    BSval = brier_score_loss(yTest, yProb)

    baseProb = np.mean(yTest)
    baseBSval = brier_score_loss(yTest, [baseProb]*len(yTest))
    BSSval = 1 - (BSval / baseBSval)

    AUCval = roc_auc_score(yTest, yProb)

    return [TPval, TNval, FPval, FNval, Pval, Nval, TPRval, TNRval, FPRval, FNRval, r, p, f1, Acc, Err, BACCval, TSSval, HSSval, SPCval, NPVval, FDRval, BSval, BSSval, AUCval]

# %% [markdown]
# *Implement 3 different classification algorithms in Python. One of them is Random Forest*  
# I have implemented the Random Forest Classifier using the Scikit-learn library.  
# 
# *The second one is from the deep learning list in the “Appendix → Additional Option: Deep Learning”*  
# I have selected and implemented the Convolutional 1D network using the Keras library.  
# 
# *The third is from the list of algorithms in “Appendix → Additional Option: Algorithms”*  
# I have selected and implemented the naive Bayes algorithm using the Scikit-learn library. This implementation comes as a result of previous attempts with an SVM taking multiple hours to train and being unable to complete training effectively.  
# 
# *This is not from scratch implementation, just use the existing libraries to implement the algorithms*  
# 
# *In evaluating classification performance, students must use the 10-fold cross validation method. You must show the statistics as discussed in the “Evaluating Classifiers”
# module to include all parameters that were introduced: TP, TF, FP, FN, TSS, HSS, etc... for each run of the 10-folds and also for overall as an average of all 10-folds execution.*  
# The getMetrics function I have written is used to calculate the statistics discussed in the "Evaluating Classifiers" module for each run of the 10 folds and also for the overall as an average of the 10 folds. Each of the three models identified are run on the same dataset.
# 
# *You must present experimental results that show the comparison of classification performance between the algorithms used in your project.*  
# The experimental results for each of the three classification methods used in this project are captured in their respective data frames for comparison.

# %%
from sklearn.ensemble import RandomForestClassifier
import keras
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

xArr = xdf.values
yArr = ydf.values

#Create calculated metrics dataframes for storage
colNames = ["True Positive", "True Negative", "False Positive", "False Negative", "Positive", "Negative", "True Positive Rate", "True Negative Rate", "False Positive Rate", "False Negative Rate", "Recall/ Sensitivity", "Precision", "F1 Score", "Accuracy", "Error Rate", "Balanced Accuracy", "True Skill Statistics", "Heidke Skill Score", "Specificity", "Negative Predictive Value", "False Discovery Rate", "Brier Score", "Brier Skill Score", "Area Under ROC Curve"]
rfcMetrics = pd.DataFrame(columns=colNames)
convMetrics = pd.DataFrame(columns=colNames)
nbMetrics = pd.DataFrame(columns=colNames)

#Set index name and start for easier viewing
rfcMetrics.index += 1
rfcMetrics.index.name = "Fold"
convMetrics.index += 1
convMetrics.index.name = "Fold"
nbMetrics.index += 1
nbMetrics.index.name = "Fold"


#implement 10-fold cross validation with stratification
k = 10
foldLenNormal = attackIndex // k
foldLenAttack = (len(xArr) - attackIndex) // k
for fold in range(k):
    print("Fold {}".format(fold+1))

    #Determine test cases in fold
    if fold < k-1:
        testIndexesNormal = range(fold * foldLenNormal, (fold + 1) * foldLenNormal)
        testIndexesAttack = range(attackIndex + fold * foldLenAttack, attackIndex + (fold + 1) * foldLenAttack)
    else:
        testIndexesNormal = range(fold * foldLenNormal, attackIndex)
        testIndexesAttack = range(attackIndex + fold * foldLenAttack, len(xArr))

    print("Testing on Normal values in {}".format(testIndexesNormal))
    print("Testing on Attack values in {}".format(testIndexesAttack))

    #Split based on fold and stratify
    #This dramatically improves the performance of Naive Bayes
    xTest = np.concatenate((xArr[testIndexesNormal], xArr[testIndexesAttack]))
    xTrain = xArr[[i for i in range(len(xArr)) if i not in testIndexesNormal and i not in testIndexesAttack]]

    yTest = np.concatenate((yArr[testIndexesNormal], yArr[testIndexesAttack]))
    yTrain = yArr[[i for i in range(len(yArr)) if i not in testIndexesNormal and i not in testIndexesAttack]]

    #Join X and Y together to shuffle
    combinedTest = np.column_stack((xTest, yTest))
    combinedTrain = np.column_stack((xTrain, yTrain))

    #Shuffle
    np.random.seed(42)
    np.random.shuffle(combinedTest)
    np.random.shuffle(combinedTrain)

    #Split X and Y test data
    xTest = combinedTest[:, :-1]
    yTest = combinedTest[:, -1]

    #Split X and Y train data
    xTrain = combinedTrain[:, :-1]
    yTrain = combinedTrain[:, -1]

    yTest = yTest.astype(int)
    yTrain = yTrain.astype(int)

    #Prior to training, the dataset is scaled to enhance performance in weight selection and convergence
    scaler = StandardScaler()
    xTrain = scaler.fit_transform(xTrain)
    xTest = scaler.transform(xTest)

    #The dimensionality is reduced to speed up execution time. With 40 components, we capture over 50% of the variance and see deminishing returns
    pca = PCA(n_components=40)
    xTrain = pca.fit_transform(xTrain)
    xTest = pca.transform(xTest)

    #Reshape to work with Conv 1D network
    xTrainSteps = xTrain.reshape((xTrain.shape[0], 1, xTrain.shape[1]))
    xTestSteps = xTest.reshape((xTest.shape[0], 1, xTest.shape[1]))

    #Create and test each model

    #Random Forest
    rfcModel = RandomForestClassifier(n_estimators=50, random_state=42)
    rfcModel.fit(xTrain, yTrain)

    #Convolutional 1D Network    
    keras.utils.set_random_seed(42)

    convModel = keras.models.Sequential()
    convModel.add(Conv1D(filters=16, kernel_size=1, activation="relu"))
    convModel.add(MaxPooling1D(pool_size=1))
    convModel.add(Conv1D(filters=32, kernel_size=1, activation="relu"))
    convModel.add(Flatten())
    convModel.add(Dense(32, activation="relu"))
    convModel.add(Dense(1, activation="sigmoid"))
    convModel.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    convModel.fit(xTrainSteps, yTrain, epochs=2, batch_size=128, validation_split=0.2)

    #Naive Bayes
    nbModel = GaussianNB()
    nbModel.fit(xTrain, yTrain)

    #Use Confusion Matrix to get TP, TN, FP, FN ONLY
    #Store results in dataframes
    rfcMetrics.loc[len(rfcMetrics.index)] = getMetrics(yTest, rfcModel.predict(xTest), rfcModel.predict_proba(xTest)[:, 1])
    convMetrics.loc[len(convMetrics.index)] = getMetrics(yTest, (convModel.predict(xTestSteps) >= 0.5).astype(int), convModel.predict(xTestSteps))
    nbMetrics.loc[len(nbMetrics.index)] = getMetrics(yTest, nbModel.predict(xTest), nbModel.predict_proba(xTest)[:, 1])
    print()

# %% [markdown]
# *You must show the statistics as discussed in the “Evaluating Classifiers” module to include all parameters that were introduced for each run of the 10-folds and also for overall as an average of all 10-folds execution*  
# The statistics calculated have been stored in the data frames for each fold and respective model, and the average for each is now calculated.

# %%
#calculate average metrics from k-fold
rfcMetrics.loc["Average"] =  rfcMetrics.mean()
convMetrics.loc["Average"] =  convMetrics.mean()
nbMetrics.loc["Average"] =  nbMetrics.mean()

# %% [markdown]
# The counts of positive and negative identifications should be whole numbers represented as integers, and are updated as such.

# %%
#convert whole numbers back to ints
intCols = ["True Positive", "True Negative", "False Positive", "False Negative", "Positive", "Negative"]
rfcMetrics[intCols] = rfcMetrics[intCols].astype(int)
convMetrics[intCols] = convMetrics[intCols].astype(int)
nbMetrics[intCols] = nbMetrics[intCols].astype(int)

# %% [markdown]
# *Provide the result of the metrics in tabular format listing all details for easier visualization (for each fold and average). Your Jupyter Notebook should also show the result in tabular format.*  
# The three sections below display the results of the metrics in a tabular format listing all details for each fold and the average for each model.

# %% [markdown]
# This section provides a tabular view of the metrics from the Random Forest Classifier.

# %%
#Set display options to prevent truncation
pd.set_option("display.max_columns", None)  #Show all columns
pd.set_option("display.expand_frame_repr", False)  #Prevent line-wrapping
pd.set_option("display.width", None) #Detect display size

print("Random Forest Classifier Performance Metrics")
print(rfcMetrics)
print()

# %% [markdown]
# This section provides a tabular view of the metrics from the Convolutional 1D network.

# %%
#Display convolutional 1D network classifier performance metrics
print("Convolutional 1D Network Classifier Performance Metrics")
print(convMetrics)
print()

# %% [markdown]
# This section provides a tabular view of the metrics from the naive Bayes algorithm.

# %%
#Display naive bayes performance metrics
print("Naive Bayes Classifier Performance Metrics")
print(nbMetrics)
print()

# %% [markdown]
# *Provide a discussion about your results. Which algorithm performs better and why? Justify your answer.*  
# The best performance metrics are identified as well as the model from which these results originated. Some metrics are optimized by selecting the maximum score, and others are optimized by selecting the minimum score. The results for all metrics are displayed for easy evaluation of the best models. A discussion of the models and results is also provided.

# %%
maxList = ["True Positive Rate", "True Negative Rate", "Recall/ Sensitivity", "Precision", "F1 Score", "Accuracy", "Balanced Accuracy", "True Skill Statistics", "Heidke Skill Score", "Specificity", "Negative Predictive Value", "Brier Skill Score", "Area Under ROC Curve"]

for col in colNames:
    if col in maxList:
        mVal = max(rfcMetrics.loc["Average", col], convMetrics.loc["Average", col], nbMetrics.loc["Average", col])
    else:
        mVal = min(rfcMetrics.loc["Average", col], convMetrics.loc["Average", col], nbMetrics.loc["Average", col])

    #Determine the model with the best value
    if mVal == rfcMetrics.loc["Average", col]:
        mModel = "Random Forest Classifier"
    elif mVal == convMetrics.loc["Average", col]:
        mModel = "1D Convolutional Network"
    elif mVal == nbMetrics.loc["Average", col]:
        mModel = "Naive Bayes Algorithm"
    else:
        mModel = "NOT FOUND"

    print("Best {}: {} = {}".format(col, mModel, mVal))

# %% [markdown]
# Based on the average results from the 10-fold cross validation testing, it is clear that the Random Forest Classifier stands out as the best algorithm due to its superior performance. The Random Forest Classifier was able to achieve the highest true positive rate, recall/sensitivity, precision, F1 score, accuracy, balanced accuracy, true skill statistics, Heidke skill score, negative predictive value, Brier skill score, and area under the ROC curve. The Random Forest Classifier was also able to achieve the lowest false negative rate, error rate, and Brier score. Based on these metrics, the Random Forest Classifier was able to outperform both the Convolutional 1D network and the naive Bayes algorithm. Within these categories, the precision, recall, F1 score, and balanced accuracy stand out by offering deeper insights into the strength of the model. The Random Forest Classifier achieved a precision of 99.89%, a recall of 99.06%, an F1 score of 99.47%, and a balanced accuracy of 99.32%. The precision of the model indicates its ability to accurately predict positive results. With this dataset, this means that fewer false positives will be generated, and thus network administrators would be more well-equipped to respond effectively and take the detected attacks seriously. If the precision were too low, network administrators may not respond with urgency to the attacks because of a high likelihood that it is a false alarm. The recall of the model indicates its ability to identify attacks and not miss threats to the network. Since the cost of missing an attack could be high, the recall gives valuable insight into the ability of the model to accurately detect if an attack is occurring within the network. The F1 score is the harmonic mean of precision and recall and provides an effective way to determine if the model is a good fit when both false positives and false negatives have a high cost. In this scenario, false positives cause network administrators to become overburdened with false alerts, resulting in a loss in ability to respond to real threats and a decreased trust in the system. False negatives have a high cost as well, making the network susceptible to exploitation and leaving critical resources vulnerable to attack. The F1 score provides valuable insight into the ability of the model to perform while both of these concerns are present. The balanced accuracy provides information about the model's ability to correctly classify data into normal and attack traffic. While the accuracy of this model is also high, the balanced accuracy provides a better understanding of the accuracy because an imbalance in the classes exists within the data, which the balanced accuracy can adjust for. The balanced accuracy of this model shows that the Random Forest Classifier is able to correctly identify attacks in the overwhelming majority of cases. These four metrics can provide a good measure of the model's performance, and help to demonstrate why the Random Forest Classifier is the best choice of the three models evaluated.  
# While the Random Forest Classifier performed the best, the Convolutional 1D network also performed incredibly well. For most metrics, the performance of the Random Forest Classifier and the Convolutional 1D network was within less than 1% difference. The Convolutional 1D network achieved a similar precision of 99.79%, a recall of 98.68%, an F1 score of 99.21%, and a balanced accuracy of 98.92%. With this dataset and the current hyperparameters for these models, the Random Forest Classifier was able to perform slightly better, but it is possible that both of these models could perform well predicting additional data from this network. Further evaluation could be done to test how well each model can generalize to the larger population, and hyperparameter optimization could enable either of these models to improve performance. These models both performed well due to the extensive features present within the data and their ability to detect complex patterns. The Random Forest Classifier and the Convolutional 1D network could be used together within the network to produce a more robust attack detection system.  
# The naive Bayes algorithm performed comparatively worse than the other two models, with a precision of 58.93%, a recall of 17.79%, an F1 score of 20.56%, and a balanced accuracy of 58.72%. This model exhibited a significantly faster training and prediction time as compared to the other two models, but struggled with certain folds of the data. This could be caused in part by the examples available in the training data within each fold and the assumption that all features are independent, which is not necessarily true with this dataset. While the correlation of the data was minimized, some network conditions could be dependent on each other. The naive Bayes model likely struggles to understand the complex patterns within the dataset due to this assumption of conditional independence.


