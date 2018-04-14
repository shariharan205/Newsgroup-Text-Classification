import matplotlib.pyplot as plt
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text

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
