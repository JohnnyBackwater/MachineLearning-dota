import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier


data=pd.read_csv('dota2Train2.csv')
data=data.drop(data.columns[[1,2,3]],axis=1)
y=data['win']
x=data.drop(data.columns[[0]],axis=1)

clf1=GaussianNB()
clf2=ExtraTreesClassifier(n_estimators=200, max_depth=None,min_samples_split=1.0, random_state=0)
clf3=RandomForestClassifier(random_state=0,n_estimators=250, min_samples_split=1.0)
clf=VotingClassifier(estimators=[('gnb',clf1), ('et', clf2),('rf',clf3)],voting='soft',weights=[4,11,1])
clf.fit(x,y)
test=pd.read_csv('dota2Test2.csv')
test=test.drop(test.columns[[1,2,3]],axis=1)
y_test=test['win']
x_test=test.drop(test.columns[[0]],axis=1)
print(clf.score(x,y))
print(clf.score(x_test,y_test))
#print(clf.feature_importances_)
