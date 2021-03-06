from TextClassification import TextClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

def breakpoint():
    print "\nContinue? [y/n] : "
    inp = raw_input()

    if inp != 'y':
        exit()

feature_selection_techniques = ["lsi", "nmf"]
df_range = [2,5]

txt_classifier_obj = TextClassifier()

print "\n============================================\n"
print "Plotting Histogram............"

ct_ra_categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                    'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

tficf_categories = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',  'misc.forsale', 'soc.religion.christian']


txt_classifier_obj.get_histogram(ct_ra_categories)

breakpoint()


print "\n============================================\n"
print "Getting TF-IDF.................."

for df in df_range:

    print "\n =============================MIN_DF - " + str(df) + " ===================================\n"
    txt_classifier_obj.get_tfidf(df=df)

    breakpoint()
    print "\n============================================\n"
    print "Running TF-ICF............."

    txt_classifier_obj.get_tficf(significant_terms_reqd = 10, target_classes = tficf_categories)


    breakpoint()
    for technique in feature_selection_techniques:


        print "\n============================================\n"
        print "\n Feature Selection Technique used - ", technique

        txt_classifier_obj.feature_selection(technique)

        breakpoint()
        print "\n============================================\n"

        print "\nHard SVM Results with gamma = 1000 with ", technique + " min_df = " + str(df)
        obj = SVC(C=1000, kernel='linear', probability=True)
        txt_classifier_obj.classify(classifier_obj=obj, classifier= "Hard SVM " + technique + " min_df = " + str(df))

        print "\nSoft SVM Results with gamma = 0.001 with ", technique + " min_df = " + str(df)
        obj = SVC(C=0.001, kernel='linear', probability=True)
        txt_classifier_obj.classify(classifier_obj=obj, classifier= "Soft SVM " + technique  + " min_df = " + str(df))

        breakpoint()
        print "\n============================================\n"

        print "\n Using 5-fold cross validation finding best penalty parameter with ", technique + " min_df = " + str(df)

        best_penalty = txt_classifier_obj.find_best_penalty(penalty_range= range(-3,4))

        print "Best penalty achieved is - ", best_penalty

        obj = SVC(C=best_penalty, kernel='linear', probability=True)
        txt_classifier_obj.classify(classifier_obj=obj, classifier = "SVM with best penalty of " + str(best_penalty) +\
                                                                    "  with " + technique  + " min_df = " + str(df))



        breakpoint()
        print "\n============================================\n"


        
        print "\nNaive Bayes Classifier with ", technique + " min_df = " + str(df)
        obj = MultinomialNB() if technique == "nmf" else GaussianNB()
        txt_classifier_obj.classify(classifier_obj=obj,
                                    classifier= "Naive Bayes " + technique  +" min_df = " + str(df))

        breakpoint()


        print "\n============================================\n"

        print "\nLogistic Regression Classifier with ", technique + " min_df = " + str(df)
        obj = LogisticRegression()
        txt_classifier_obj.classify(classifier_obj=obj,
                                    classifier= "Logistic Regression " + technique  + " min_df = " + str(df))

        breakpoint()
        
        print "\n============================================\n"

        print "\nLogistic Regression Classifier with L1 regularization with ", technique + " min_df = " + str(df)
        obj = LogisticRegression(penalty="l1")
        txt_classifier_obj.classify(classifier_obj=obj,
                                    classifier= "Logistic Regression L1 " + technique  + " min_df = " + str(df))



        print "\nLogistic Regression Classifier with L2 regularization with ", technique + " min_df = " + str(df)
        obj = LogisticRegression(penalty="l2")
        txt_classifier_obj.classify(classifier_obj=obj,
                                    classifier= "Logistic Regression L2 " + technique + " min_df = " + str(df))
 
        breakpoint()
        
        print "Testing different regularization parameters for logistic regression"

        for regularization in ["l1", "l2"]:

            print "Regularization type - ", regularization

            for c in range(-3,4):

                print "Inverse of regularization C = ", 10**c
                breakpoint()
                obj = LogisticRegression(penalty=regularization, C=10**c)
                txt_classifier_obj.classify(classifier_obj=obj, plot_roc=False, get_coef=True)

    breakpoint()


print "\n============================================\n"

txt_classifier_obj.get_histogram(tficf_categories, plot = False)

for df in df_range:

    print "\n =============================MIN_DF - " + str(df) + " ===================================\n"

    txt_classifier_obj.get_tfidf(df=df)

    for technique in feature_selection_techniques:

        print "\n Feature Selection Technique used - ", technique + " min_df = " + str(df)

        txt_classifier_obj.feature_selection(technique)

        print "Multiclass Naive Bayes with ", technique + " min_df = " + str(df)
        obj = MultinomialNB() if technique == "nmf" else GaussianNB()
        txt_classifier_obj.classify(classifier_obj=obj,lower_threshold = False, plot_roc = False,
                                    classifier= "Multiclass NaiveBayes " + technique + " min_df = " + str(df))

        breakpoint()

        print "One Vs One with SVM with ", technique + " min_df = " + str(df)
        obj = OneVsOneClassifier(SVC(kernel='linear', probability=True))
        txt_classifier_obj.classify(classifier_obj=obj,lower_threshold = False, plot_roc = False,
                                    classifier= "1vs1 SVM " + technique + " min_df = " + str(df))

        print "One Vs Rest with SVM with ", technique + " min_df = " + str(df)
        obj = OneVsRestClassifier(SVC(kernel='linear', probability=True))
        txt_classifier_obj.classify(classifier_obj=obj,lower_threshold = False, plot_roc = False,
                                    classifier= "1vsRest SVM " + technique + " min_df = " + str(df))
                                    
        breakpoint()

