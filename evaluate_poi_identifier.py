#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson13_keys.pkl')
labels, features = targetFeatureSplit(data)



### your code goes here
from sklearn.model_selection import train_test_split
feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size= 0.3, random_state=42)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(feature_train, label_train)
print(clf.score(feature_test, label_test))

#understand prediction result, there is certainly an imbalance
pred = clf.predict(feature_test)
print(sum(pred))
print(len(feature_test))
print(sum(label_test))
print(sum(pred * label_test))

#evaluate precision and recall
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print("precision: ", precision_score(label_test, pred))
print("recall: ", recall_score(label_test, pred))
