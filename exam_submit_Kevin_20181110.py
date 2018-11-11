import numpy
import numpy as np
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

#Print on/off toggle for debugging purpose. L
DPrint = False

# Reading TSV file
fname = 'AP_ICD10.tsv'
dataset = read_csv(fname, sep='\t')

# Basic study on data
# shape
if DPrint: print(dataset.shape)
# types
set_option('display.max_rows', 500)
if DPrint: print(dataset.dtypes)
# head
set_option('display.width', 300)
if DPrint: print(dataset.head(20))

###################################################
#ML learning Data Preprocessing
#select AP column (Column # 1) for X and ICD10 code (Column #2) for Y
##################################################
X = dataset.iloc[:,1].astype(str)
Y_temp = dataset.iloc[:,2].astype(str)
Y = Y_temp

# Take the first 3 chars of the ICD10 codes
for i in range(len(Y_temp)):
    Y[i] = Y_temp[i][:3]

if DPrint: print(X.head(20))
if DPrint: print(Y.head(20))

#################################################################################
# Convert the text files into numerical feature vectors using bag of words model
# create feature vectors using 'CountVectorizer' and frequency count &
# reduce common words using TF-IDF
#################################################################################

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()

# Pipelining for data transformation
datapipeline = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer())])

# Train & Test split by the ratio of 20%
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

# This may be used for some other purposes.
#X_train_transformed = datapipeline.fit_transform(X_train, Y_train)
#X_validation_transformed = datapipeline.fit_transform(X_validation, Y_validation)

# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'accuracy'

# Standardize the dataset for different Algorithms:
pipelines = []
pipelines.append(('TransformedLR', Pipeline([('DataP', datapipeline),('LR', LogisticRegression())])))
pipelines.append(('TransformeddKNN', Pipeline([('DataP', datapipeline),('KNN', KNeighborsClassifier())])))
pipelines.append(('TransformedCART', Pipeline([('DataP', datapipeline),('CART', DecisionTreeClassifier())])))
pipelines.append(('TransformedSVM', Pipeline([('DataP', datapipeline),('SVM', SVC())])))
pipelines.append(('TransformedMNB', Pipeline([('DataP', datapipeline),('MNB', MultinomialNB())])))
pipelines.append(('TransformedSVM2', Pipeline([('DataP', datapipeline),('SVM2', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])))

results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

###########################################################################################################
# From the different algorithms, SGDClassifier gives the best result
# So, perform paramter tuning for that using GridsearchCV on ngram_range, use_idf, and Alpha for SGDClassifier
##########################################################################################################
from sklearn.model_selection import GridSearchCV

finalrun_svm2 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])
parameters_svm2 = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),'clf-svm__alpha': (1e-2, 1e-3)}
orun= finalrun_svm2
kfold = KFold(n_splits=num_folds, random_state=seed)
gs_clf_svm2 = GridSearchCV(orun, parameters_svm2, n_jobs=None, scoring=scoring, cv=kfold)
gs_clf_svm2 = gs_clf_svm2.fit(X_validation, Y_validation)
print("GridSearchCV parameter study results and parameter set")
print(gs_clf_svm2.best_score_)
print(gs_clf_svm2.best_params_)
predicted = gs_clf_svm2.predict(X_validation)

print("#####################################################")
print("Report: result on Validation set (20% of the data)")
print(np.mean(predicted == Y_validation))


############Report##############################################################
# SGDClassifier model was made with parameters {'vect__ngram_range': (1, 1), 'tfidf__use_idf': True, 'clf-svm__alpha': 0.01}
# The run on the X_validation (20% of the given data set)
# The prediction accuracy is found to be 96%
################################################################################
