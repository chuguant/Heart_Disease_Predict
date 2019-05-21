import pandas as pd
import numpy as np


column_names = [
    'age',
    'sex',
    'chest pain type',
    'resting blood pressure',
    'serum cholestoral',
    'fasting blood sugar',
    'resting electrocardiographic results',
    'maximum heart rate',
    'angina',
    'oldpeak = ST depression',
    'the slope of the peak exercise ST segment',
    'number of major vessels (0-3) colored by flourosopy',
    'thal',
    'target'
]


df = pd.read_csv('./processed.cleveland.data', header=None,
                 na_values='?', names=column_names)

df = df.dropna()

df.loc[df['target'] != 0, 'target'] = 1

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# clf = LogisticRegression(solver='liblinear').fit(X_train, y_train)
# clf.score(X_test, y_test)
# clf.score(X_train, y_train)

# y_pred = clf.predict(X_test)

from sklearn.ensemble import BaggingClassifier

# Logistic Regression
clf = LogisticRegression(solver='liblinear')
scores = cross_val_score(clf, X, y, cv=5)
print("LR: {:0.4f} (+/- {:0.2f})".format(scores.mean(), scores.std() * 2))

bagging = BaggingClassifier(clf, max_samples=0.5, max_features=0.5)
scores = cross_val_score(bagging, X, y, cv=5)
print("LR bagging: {:0.4f} (+/- {:0.2f})".format(scores.mean(), scores.std() * 2))

scores = cross_val_score(clf, X_norm, y, cv=5)
print("LR norm: {:0.4f} (+/- {:0.2f})".format(scores.mean(), scores.std() * 2))

# SVM
from sklearn import svm
clf = svm.SVC(gamma='scale')
scores = cross_val_score(clf, X, y, cv=5)
print("SVM: {:0.4f} (+/- {:0.2f})".format(scores.mean(), scores.std() * 2))
scores = cross_val_score(clf, X_norm, y, cv=5)
print("SVM norm: {:0.4f} (+/- {:0.2f})".format(scores.mean(), scores.std() * 2))

# Decision Tree
from sklearn import tree
clf = tree.DecisionTreeClassifier()
scores = cross_val_score(clf, X, y, cv=5)
print("Decision Tree: {:0.4f} (+/- {:0.2f})".format(scores.mean(), scores.std() * 2))
scores = cross_val_score(clf, X_norm, y, cv=5)
print("Decision Tree norm: {:0.4f} (+/- {:0.2f})".format(scores.mean(), scores.std() * 2))

# Adaboost
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(clf, X, y, cv=5)
print("Adaboost: {:0.4f} (+/- {:0.2f})".format(scores.mean(), scores.std() * 2))
scores = cross_val_score(clf, X_norm, y, cv=5)
print("Adaboost norm: {:0.4f} (+/- {:0.2f})".format(scores.mean(), scores.std() * 2))

# Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2, criterion='entropy')
scores = cross_val_score(clf, X, y, cv=5)
print("Random Forest: {:0.4f} (+/- {:0.2f})".format(scores.mean(), scores.std() * 2))

# for criterion in ('gini', 'entropy'):
#     for n_estimators in range(10, 150, 10):
#         for max_depth in (None, 2, 3, 4, 5, 6):
#             clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion)
#             scores = cross_val_score(clf, X, y, cv=5)
#             print("{}-{}-{}: {:0.4f} (+/- {:0.2f})".format(criterion, n_estimators, max_depth, scores.mean(), scores.std() * 2))

# KNN
# from sklearn.neighbors import NearestNeighbors
# for k in range(2, 10):
#     clf = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
#     scores = cross_val_score(clf, X, y, cv=5)
#     print('KNN {}:'.format(k), np.mean(scores))
#     scores = cross_val_score(clf, X_norm, y, cv=5)
#     print('KNN {} norm:'.format(k), np.mean(scores))

# import pickle
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=100, max_depth=2, criterion='entropy')
# clf.fit(X, y)
# y_pred = clf.predict(X)
# print("Random Forest train accuracy: {:0.4f}".format(np.sum(y_pred==y)/len(y)))

# pickle.dump(clf, open('random_forest.pickle', 'wb'))
