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

