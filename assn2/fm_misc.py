from sklearn.metrics import roc_auc_score

preds = [] 
with open("fm_pred.txt", "r") as f: 
	for line in f: 
		proba = float(line) 
		preds.append(proba) 

