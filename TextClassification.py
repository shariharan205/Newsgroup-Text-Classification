import numpy as np
import matplotlib.pyplot as plt
from nltk.stem.porter import *
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
from sklearn import metrics, cross_validation
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

def breakpoint():
    """

    Allows the flow of program based on the control of the user.
    """
    print "\nContinue? [y/n] : "
    inp = raw_input()

    if inp != 'y':
        exit()



class TextClassifier(object):

    def get_data(self, category_list = [], subset_var = 'train'):
        """
        Takes a category list and a flag whether train/test set has to be retrieved and returns
        the corresponding data from 20newsgroups. If no list is passed, all data is returned.
        """

        if not category_list:
            return fetch_20newsgroups(subset=subset_var, random_state=42, shuffle=True)

        return fetch_20newsgroups(subset=subset_var, categories=category_list, random_state=42, shuffle=True)


    def get_stop_words(self):
        """
        Returns a list of stop words
        """
        #return set(stopwords.words('english')) #contains only 153 stop words
        return text.ENGLISH_STOP_WORDS #contains 318 stop words


    def get_stemmer(self):
        """
        Snowball stemmer in English performs better than PorterStemmer
        Example : stem("generously") with Porter gives "gener" while SnowballStemmer gives "generous"
        """
        #return PorterStemmer()
        return SnowballStemmer("english")


    def get_significant_terms(self, row, features, number_of_terms):
        """
        Returns the significant terms by zipping and sorting the (tficf_value, feature_name) tuple
        """
        return zip(*sorted(zip(row[0], features),reverse=True)[:number_of_terms])[1]

    def threshold(self, data, lower_threshold):
        """
        Separates the values with index less than or equal to threshold to be 0 and the rest to be 1.
        """
        return map(lambda x : int(not x <= lower_threshold), data) if lower_threshold else data


    def plot_ROC(self, y_test, y_score, classifier = ""):
        """
        Gets the testing target and corresponding probability score of features as input.
        False positive rate and True positive rate and obtained from the roc_curve method and the figure is plotted
        """

        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_score)

        plt.figure()
        plt.plot(false_positive_rate, true_positive_rate, lw = 2, label= 'area under curve = %0.4f' % auc(false_positive_rate,
                                                                                                          true_positive_rate))
        plt.grid(color='0.7', linestyle='--', linewidth=1)
        plt.xlabel('False Positive Rate', fontsize=15)
        plt.ylabel('True Positive Rate', fontsize=15)
        plt.legend(loc="lower right")
        plt.title('ROC Curve for ' + classifier)
        plt.savefig(classifier + '.png', format='png')
        plt.show()


    def preprocess(self, txt):
        """
        Takes a string and preprocesses it by:
            1. Removing all characters other than alphabets.
            2. Split the string into words
            3. Removing the stop words
            4. Removing the words with length less than 3
            5. Stem the words with lower case
            6. Join the list of words back to a sentence.
        """

        txt = re.sub(r'[^a-zA-Z]+', ' ', txt)
        words = txt.split()
        stop_words = self.get_stop_words()
        words = [self.get_stemmer().stem(word.lower()) for word in words if word not in stop_words and len(word) > 2]
        return ' '.join(words)


    def get_histogram(self, categories, plot = True):
        """
        Given the list of categories, retrieve the newsgroup data and plot the histogram.
        We use a Counter to get the words and corresponding frequency counts
        """

        self.train = self.get_data(categories, subset_var= 'train')
        self.test = self.get_data(categories, subset_var='test')
        counts = Counter(self.train.target)
        frequencies = [counts[i] for i in counts]

        if plot:
            plt.bar(self.train.target_names, frequencies, 0.5, color='b')
            plt.title("#Training docs per class histogram")
            plt.xlabel("Class")
            plt.ylabel("Number of training documents")
            plt.xticks(fontsize = 8, rotation=20)
            plt.savefig('histogram.png', format='png')
            plt.show()

        print "Total number of Computer Technology training documents = " , sum(frequencies[:4])
        print "Total number of Recreational Activity training documents = ", sum(frequencies[4:])
        print "Mean of training documents size in these 8 categories = ", np.mean(frequencies)
        print "Standard deviation = ", np.std(frequencies)




    def get_tfidf(self, df = 2):
        """
        Training and testing data are preprocessed.
        Countvectorizer and TfidfTransformer are used to extract TF-IDF features.
        """

        for i in range(len(self.train.data)):
            self.train.data[i] = self.preprocess(self.train.data[i])


        for i in range(len(self.test.data)):
            self.test.data[i] = self.preprocess(self.test.data[i])

        cv = CountVectorizer(min_df=df)
        tf = TfidfTransformer()

        train_counts = cv.fit_transform(self.train.data)
        self.tfidf_train = tf.fit_transform(train_counts)

        test_counts = cv.transform(self.test.data)
        self.tfidf_test = tf.transform(test_counts)

        print "Dimension of TF-IDF features for training dataset with min_df = ", df, " : ", self.tfidf_train.shape
        print "Dimension of TF-IDF features for testing dataset with min_df = ", df, " : ", self.tfidf_test.shape


    def get_tficf(self, significant_terms_reqd, target_classes):
        """
        Here, the same process as TF-IDF is done, but we keep summing each row of a tficf matrix
        with the count frequency of corresponding index.  We use feature_names() to map the row number
        to the word name
        """

        all_category_data = self.get_data()
        all_categories = all_category_data.target_names
        n_classes = len(all_categories)

        for i in range(len(all_category_data.data)):
            all_category_data.data[i] = self.preprocess(all_category_data.data[i])

        count_vec = CountVectorizer(min_df=5)
        count_frequency = count_vec.fit_transform(all_category_data.data)
        docs_len, terms_len = count_frequency.shape

        tficf = np.zeros(shape=(n_classes, terms_len))

        for i in range(docs_len):
            category_index = all_category_data.target[i]
            tficf[category_index,] = tficf[category_index,] + count_frequency[i,]

        tficf = TfidfTransformer(use_idf=True).fit(count_frequency).transform(tficf)

        features = count_vec.get_feature_names()

        print "\nClass    -       Top 10 significant terms for the class"
        for class_name in target_classes:
            print class_name, "-" , self.get_significant_terms(tficf[all_categories.index(class_name)].toarray(),
                                                               features, significant_terms_reqd)




    def feature_selection(self, technique = "lsi"):
        """
        Performs dimension reduction to 50 components on lsi and nmf based on the input argument.
        """

        if technique == "lsi":

            print '\nThe dimensions before performing LSI are ' , self.tfidf_train.shape
            svd = TruncatedSVD(n_components=50)
            self.train_features = svd.fit_transform(self.tfidf_train)
            print 'The dimensions after performing LSI are ' , self.train_features.shape

            self.test_features = svd.transform(self.tfidf_test)

        elif technique == "nmf":

            print '\nThe dimensions before performing NMF are ' , self.tfidf_train.shape
            nmf_obj = NMF(n_components=50, init='random', random_state=0)
            self.train_features = nmf_obj.fit_transform(self.tfidf_train)
            print 'The dimensions after performing NMF are ' , self.train_features.shape

            self.test_features = nmf_obj.transform(self.tfidf_test)


    def classify(self, classifier_obj = None, classifier = "", lower_threshold = 3, plot_roc = True, get_coef=False):
        """
        We convert the target from multiclass to binary by grouping together the classes.

        We then fit the training data to the model object and predict based on the test set.
        Also, this method prints the metrics like accuracy, precision recall, confusion matrix and ROC plot.
        """


        #first 4 values set to 0 - Computer Technology, next 4 set to 1 - Recreational Activity
        self.train_target = self.threshold(self.train.target, lower_threshold = lower_threshold)
        self.test_target  = self.threshold(self.test.target, lower_threshold = lower_threshold)

        obj = classifier_obj.fit(self.train_features, self.train_target)
        result = obj.predict(self.test_features)

        print "\nAccuracy  - " , metrics.accuracy_score(self.test_target, result) * 100, '%'

        print "\nPrecision and Recall values: \n", metrics.classification_report(self.test_target , result)
        print "\nConfusion matrix:\n"
        print metrics.confusion_matrix(self.test_target , result)

        if plot_roc:
            self.plot_ROC(self.test_target, obj.predict_proba(self.test_features)[:,1], classifier)

        if get_coef:
            #print "Co-efficient matrix", obj.coef_
            print "Mean co-efficient - ", np.mean(obj.coef_)



    def find_best_penalty(self, penalty_range):
        """
        This method performs k-fold cross validation and finds the best penalty parameter.
        """

        max_score = -float("inf")
        best_penalty = None

        for penalty in penalty_range:
            obj = SVC(C = 10**penalty, kernel='linear', probability=True)
            classifier_obj = obj.fit(self.train_features, self.train_target)
            result = classifier_obj.predict(self.test_features)

            print "Accuracy for penalty = ", penalty , " is : ", metrics.accuracy_score(self.test_target, result) * 100, '%'
            cv_score = np.mean(cross_validation.cross_val_score(obj, self.train_features, self.train_target, cv=5))
            if cv_score > max_score:
                max_score = cv_score
                best_penalty = penalty

        return 10**best_penalty


