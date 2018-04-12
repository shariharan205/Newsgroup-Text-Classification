# Newsgroup-Text-Classification

In this project, “20 Newsgroups” dataset is used. It is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups, each corresponding to a different topic. For majority of the classification tasks, 8 sub classes of two major classes 'Computer Technology' and 'Recreational Activity' are used.

Subclasses of Computer Technology - comp.graphics, comp.os.ms-windows.misc, comp.sys.ibm.pc.hardware, comp.sys.mac.hardware
Subclasses of Recreational Activity - rec.autos, rec.motorcycles, rec.sport.baseball, rec.sport.hockey

The documents of balanced data of 8 classes, are converted into numerical feature vectors. Each document is tokenized into words and the stop words and punctuations are excluded, and 
finally the stemmed version of words are used. These are then used to create a Term Frequency - Inverse Document Frequency (TF-IDF) vector representation. For feature selection, the performance of Latent Semantic Indexing (LSI) and Non-negative Matrix Factorization (NMF) are compared. 

For classifiers, Support Vector Machines, Naive Bayes Classifier and Logistic Regression Classifier are experimented.

TextClassification.py file essentially contains all the data preprocessing, feature extraction, dimension reduction, classification, etc.
run.py gets the control from the user and passes it on to the TextClassification class. 

To run the project:

python run.py 
	
After completing each sub task, the execution pauses and asks for the user to continue or not. Press y to continue. 

