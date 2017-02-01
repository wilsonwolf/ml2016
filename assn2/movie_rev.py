import pandas as pd 
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

movie_tr = pd.read_csv("MovieReview_train.csv") 
X_tr = movie_tr.drop('sentiment', 1) 
y_tr = movie_tr['sentiment'] 

movie_test = pd.read_csv("MovieReview_test.csv") 


max_depths = list(range(1, 4) ) 
estimators = [] 
for d in max_depths: 
	estimators.append( DecisionTreeClassifier(max_depth = d) ) 

ada_params = {'base_estimator': estimators,
	'n_estimators': list(range(50, 200, 10)), 
	'learning_rate': [0.01]  
	} 
rfc_params = {'n_estimators': list(range(50, 200, 10)), 
	'max_depth': list(range(1, 5))
	}

gb_params = {'n_estimators': list(range(50, 200, 10)), 
	'max_depth': list(range(1,5)) 
	} 

ada_clf = GridSearchCV( AdaBoostClassifier(), ada_params, scoring = "accuracy", cv = 10)
rfc_clf= GridSearchCV( RandomForestClassifier(), rfc_params, scoring = "accuracy", cv = 10) 
gb_clf = GridSearchCV( GradientBoostingClassifier(), gb_params, scoring = "accuracy", cv = 10)

ada_clf.fit(X_tr, y_tr) 