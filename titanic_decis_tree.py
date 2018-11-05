import pandas
from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv('data/titanic.csv', index_col='PassengerId')

info = data.filter(items=['Pclass', 'Fare', 'Age', 'Sex', 'Survived']).dropna()
info = info.replace(to_replace=['male', 'female'], value=[1, 0])

features = info.filter(items=['Pclass', 'Fare', 'Age', 'Sex'])
survived = info['Survived']

clf = DecisionTreeClassifier(random_state=241)
clf.fit(features, survived)
importances = clf.feature_importances_
print(importances)
