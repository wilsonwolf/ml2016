#Author: Karl Jiang 
#Script to convert pandas dataframe into lib format (Will be used for libFM) 

import pandas as pd 
from scipy.sparse import csr_matrix 
from sklearn.model_selection import StratifiedKFold

'''
params: 
	df - panda dataframe 
	colname - name of the column that is your dependent variable 
Returns tuple X, y
	- X is a matrix like object of covariate data 
	- y is array of responses
'''
def X_y_split(df, colname): 
	X = df.drop(colname, 1) 
	y = df[colname] 
	return (csr_matrix(X), y) 

def create_line(line, X, i):
	data = X.data
	indptr = X.indptr 
	indices = X.indices
	
	col_indices = indices[indptr[i]:indptr[i + 1]]
	vals = data[indptr[i]:indptr[i + 1] ]

	for c, v in zip(col_indices, vals): 
		line = line + str(c) + ":" + str(v) + " " 
	return line  

def conv_train(X, y, filename = "default_name.txt"): 
	with open(filename, "w") as f:
		for i in range(0, X.shape[0] ): 
			l = str(y[i]) + " "
			line = create_line(l, X, i) 
			f.write(line + "\n") 
				
def conv_test(X, filename = "default_name_test.txt"): 
	y = [0] * X.shape[0] 
	conv_train(X, y, filename) 		
	
		

if __name__ == "__main__": 
	movie_tr = pd.read_csv("MovieReview_train.csv") 

	skf = StratifiedKFold(n_splits=2)		
	X, y = X_y_split(movie_tr, "sentiment") 
	conv_train(X, y, "movie_train.txt") 

	movie_test = pd.read_csv("MovieReview_test.csv") 
	X_test = csr_matrix(movie_test)
	conv_test(X_test, "movie_test.txt")  

		


