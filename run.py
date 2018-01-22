# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

k_fold = KFold(n_splits=12, shuffle=True, random_state=0)


def bar_chart(feature):
    all_ps = train_data[feature].value_counts()
    survived = train_data[train_data['Survived']==1][feature].value_counts()
    dead = train_data[train_data['Survived']==0][feature].value_counts()
    df = pd.DataFrame([all_ps, survived,dead])
    df.index = ['All', 'Survived','Dead']
    df.plot(kind='bar',stacked=False, figsize=(10,5))
    
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

#Sex category
train_data['Sex'] = train_data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
test_data['Sex'] = test_data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

#Embarked
train_data['Embarked'] = train_data['Embarked'].map( {'S': 0, 'Q': 1, 'C': 2} )
train_data['Embarked'] = train_data['Embarked'].fillna(value = 3)

test_data['Embarked'] = test_data['Embarked'].map( {'S': 0, 'Q': 1, 'C': 2} )
test_data['Embarked'] = test_data['Embarked'].fillna(value = 3)


#categorize Title
train_data.loc[train_data['Name'].str.contains("Mr."), 'Title'] = 1
train_data.loc[train_data['Name'].str.contains("Mrs."), 'Title'] = 2
train_data.loc[train_data['Name'].str.contains("Miss."), 'Title'] = 3
train_data.loc[train_data['Name'].str.contains("Lady."), 'Title'] = 3
train_data.loc[train_data['Name'].str.contains("Ms."), 'Title'] = 3
train_data['Title'] = train_data['Title'].fillna(value = 4)

test_data.loc[test_data['Name'].str.contains("Mr."), 'Title'] = 1
test_data.loc[test_data['Name'].str.contains("Mrs."), 'Title'] = 2
test_data.loc[test_data['Name'].str.contains("Miss."), 'Title'] = 3
test_data.loc[test_data['Name'].str.contains("Lady."), 'Title'] = 3
test_data.loc[test_data['Name'].str.contains("Ms."), 'Title'] = 3
test_data['Title'] = test_data['Title'].fillna(value = 4)


#fill missing age
train_data["Age"].fillna(train_data.groupby("Title")["Age"].transform("median"), inplace=True)
test_data["Age"].fillna(test_data.groupby("Title")["Age"].transform("median"), inplace=True)
#category age (young, adult, senior)train_data.loc[train_data['Age'] <= 16, 'Age'] = 1
train_data.loc[(train_data['Age'] > 16) & (train_data['Age'] <= 59), 'Age'] = 2
test_data.loc[(test_data['Age'] > 59) & (test_data['Age'] <= 64), 'Age'] = 3
train_data.loc[train_data['Age'] > 64, 'Age'] = 4

test_data.loc[test_data['Age'] <= 16, 'Age'] = 1
test_data.loc[(test_data['Age'] > 16) & (test_data['Age'] <= 59), 'Age'] = 2
test_data.loc[(test_data['Age'] > 59) & (test_data['Age'] <= 64), 'Age'] = 3
test_data.loc[test_data['Age'] > 64, 'Age'] = 4

#Cabin
train_data.loc[train_data['Cabin'].str.contains("A", na=False), 'Cabin'] = 1
train_data.loc[train_data['Cabin'].str.contains("B", na=False), 'Cabin'] = 2
train_data.loc[train_data['Cabin'].str.contains("C", na=False), 'Cabin'] = 3
train_data.loc[train_data['Cabin'].str.contains("D", na=False), 'Cabin'] = 4
train_data.loc[train_data['Cabin'].str.contains("E", na=False), 'Cabin'] = 5
train_data.loc[train_data['Cabin'].str.contains("F", na=False), 'Cabin'] = 6
train_data.loc[train_data['Cabin'].str.contains("G", na=False), 'Cabin'] = 7
train_data.loc[train_data['Cabin'].str.contains("T", na=False), 'Cabin'] = 8
train_data['Cabin'] = train_data['Cabin'].fillna(value = 9)

test_data.loc[test_data['Cabin'].str.contains("A", na=False), 'Cabin'] = 1
test_data.loc[test_data['Cabin'].str.contains("B", na=False), 'Cabin'] = 2
test_data.loc[test_data['Cabin'].str.contains("C", na=False), 'Cabin'] = 3
test_data.loc[test_data['Cabin'].str.contains("D", na=False), 'Cabin'] = 4
test_data.loc[test_data['Cabin'].str.contains("E", na=False), 'Cabin'] = 5
test_data.loc[test_data['Cabin'].str.contains("F", na=False), 'Cabin'] = 6
test_data.loc[test_data['Cabin'].str.contains("G", na=False), 'Cabin'] = 7
test_data.loc[test_data['Cabin'].str.contains("T", na=False), 'Cabin'] = 8
test_data['Cabin'] = test_data['Cabin'].fillna(value = 9)

#family size
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

#category Fare (economic vs luxuary)
train_data['Fare'] = train_data['Fare'] / train_data['FamilySize']
train_data.loc[train_data['Fare'] <= 9, 'Fare'] = 1
train_data.loc[train_data['Fare'] > 9, 'Fare'] = 2

test_data['Fare'] = test_data['Fare'] / test_data['FamilySize']
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())
test_data.loc[test_data['Fare'] <= 9, 'Fare'] = 1
test_data.loc[test_data['Fare'] > 9, 'Fare'] = 2

#drop Name, ticket, Parch and SibSp
y_train = train_data['Survived'].ravel()

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
x_train = train_data.drop(drop_elements, axis = 1)

x_train = x_train.drop(['Survived'], axis=1)


passengerIds = test_data["PassengerId"]

x_test = test_data.drop(drop_elements, axis = 1)


#clf = RandomForestClassifier(n_estimators=11)
#clf.fit(x_train, y_train)
#prediction = clf.predict(x_test)
#result_rf = pd.DataFrame({
#        "PassengerId": passengerIds,
#        "Survived": prediction
#    })
#    
#result_rf.to_csv('submission_rf.csv', index=False)
#result_rf.Survived.value_counts().plot(kind="bar")    
#
#clf = SVC()
#clf.fit(x_train, y_train)
#prediction_svm = clf.predict(x_test)
#result_svm = pd.DataFrame({
#        "PassengerId": passengerIds,
#        "Survived": prediction
#    })
#    
#result_svm.to_csv('submission_svm.csv', index=False)
#result_svm.Survived.value_counts().plot(kind="bar")  

clf = RandomForestClassifier(n_estimators=11)    
scoring = 'accuracy'
score = cross_val_score(clf, x_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, x_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


clf = SVC()
clf.fit(x_train, y_train)
prediction = clf.predict(x_test)
submission = pd.DataFrame({
        "PassengerId": passengerIds,
        "Survived": prediction
    })

submission.to_csv('submission.csv', index=False)






