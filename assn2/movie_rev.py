import pandas as pd 
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split


movie_tr = pd.read_csv("MovieReview_train.csv") 
X_tr = movie_tr.drop('sentiment', 1) 
y_tr = movie_tr['sentiment'] 

movie_test = pd.read_csv("MovieReview_test.csv") 


max_depths = list(range(2, 4) ) 
estimators = [] 
for d in max_depths: 
	estimators.append( DecisionTreeClassifier(max_depth = d) ) 

ada_params = {'base_estimator': estimators,
	'n_estimators': list(range(200, 300, 10)), 
	'learning_rate': [0.01]  
	} 
rfc_params = {'n_estimators': list(range(201, 219, 2)), 
	'max_depth': list(range(2, 12, 2))
	}

gb_params = {'n_estimators': list(range(240, 260, 2)), 
	'max_depth': list(range(3,7)),  
	'learning_rate': [0.01, 0.1] 
	} 

"""
ada_clf = GridSearchCV( AdaBoostClassifier(), ada_params, scoring = "roc_auc", cv = 10)
rfc_clf= GridSearchCV( RandomForestClassifier(), rfc_params, scoring = "roc_auc", cv = 10) 
gb_clf = GridSearchCV( GradientBoostingClassifier(), gb_params, scoring = "roc_auc", cv = 10)

ada_clf.fit(X_tr, y_tr) 
pd.to_pickle(ada_clf, "ada_gridsearch.pkl") 

rfc_clf.fit(X_tr, y_tr) 
pd.to_pickle(rfc_clf, "rfc_gridsearch.pkl") 

gb_clf.fit(X_tr, y_tr) 
pd.to_pickle(gb_clf, "gb_gridsearch.pkl") 
""" 

ada_clf = pd.read_pickle("ada_gridsearch.pkl") 
rfc_clf = pd.read_pickle("rfc_gridsearch.pkl")
gb_clf = pd.read_pickle("gb_gridsearch.pkl")

X_train, X_test, y_train, y_test = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)
rfc_best = rfc_clf.best_estimator_ 
rfc_best.fit(X_train, y_train) 
rfc_pred = rfc_best.predict(X_test) 
res = sum(rfc_pred == y_test) / float(len(rfc_pred) ) 

