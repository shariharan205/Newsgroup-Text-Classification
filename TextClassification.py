import numpy as np
import matplotlib.pyplot as plt
from nltk.stem.porter import *
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
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


