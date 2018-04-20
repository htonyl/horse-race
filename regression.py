import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def validate(pred_result , true_labels, df):
	rmse = np.sqrt(mean_squared_error(pred_result, true_labels))
	aggreg  = df.join(pd.DataFrame({'pred_time': pred_result}))
	total = 0
	top_1 = 0
	top_3 = 0
	ave_rank = 0
	for name, eachrace in aggreg.groupby('race_id'):
		total += 1
		Top1_Rank = eachrace.nsmallest(1,'pred_time')['finishing_position'].iloc[0]
		if Top1_Rank == 1:
			top_1 += 1
		if Top1_Rank <= 3:
			top_3 += 1
		ave_rank += Top1_Rank
	return (rmse, float(top_1)/total, float(top_3)/total, float(ave_rank)/total)

def extract(df):
	features = df.as_matrix(columns=['actual_weight', 'declared_horse_weight', 'draw','win_odds', 'jockey_ave_rank', 'trainer_ave_rank', 'recent_ave_rank', 'race_distance'])
	labels = df.as_matrix(columns=['finish_time'])
	for i in range(len(labels)):
		min,sec,msec = labels[i][0].split('.')
		labels[i][0] = float(min)*60*60 + float(sec) * 60 + float(msec)
	labels = np.reshape(labels,-1)
	return features, labels
# Read
fname = './training.csv'
print('Reading dataframe from ', fname)
print('\n')
train_data = pd.read_csv(fname)



fname = './testing.csv'
print('Reading dataframe from ', fname)
print('\n')
test_data = pd.read_csv(fname)

train_features , train_labels = extract(train_data)
test_features , true_labels = extract(test_data)

#4.1.1.
svr_model = SVR(kernel='linear')
svr_model.fit(train_features,train_labels)
svr_pred = svr_model.predict(test_features)

#4.1.2.
#gbrt_model = GradientBoostingRegressor()
#gbrt_model.fit(train_features,train_labels)
#gbrt_pred = gbrt_model.predict(test_features)

svr_stat = ("svr_model", ) + validate(svr_pred, true_labels, test_data[['race_id', 'finishing_position']])
print(svr_stat)
#gbrt_stat = ("gbrt_model", )+ validate(gbrt_pred, true_labels, test_data[['race_id','finishing_position']])
