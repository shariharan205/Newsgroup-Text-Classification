from nltk.stem.snowball import SnowballStemmer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text

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


