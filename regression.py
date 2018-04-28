import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import ctypes
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
		eachrace = eachrace.sample(frac=1).reset_index(drop=True)
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


mkl_rt = ctypes.CDLL('libmkl_rt.so')
mkl_get_max_threads = mkl_rt.mkl_get_max_threads
print(mkl_get_max_threads)
def mkl_set_num_threads(cores):
	mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))

mkl_set_num_threads(4)
print( mkl_get_max_threads())

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

feat_scaler = StandardScaler()
label_scaler = StandardScaler()
norm_train_feats = feat_scaler.fit_transform(train_features)
norm_test_feats = feat_scaler.transform(test_features)

norm_train_labels = label_scaler.fit_transform(train_labels.reshape(-1,1)).reshape(-1)


#4.1.1.
#epsilon=0.2
svr_param_space = {"C":[0.01, 0.05,0.2,0.5,1,2],
		"epsilon":[0.01,0.05,0.15, 0.2]
			}
log = []
for c in svr_param_space["C"]:
	for eps in svr_param_space["epsilon"]:
		print("\n")
		norm_svr_model = SVR(cache_size=6000,kernel='poly',C=c, epsilon=eps)
		norm_svr_model.fit(norm_train_feats,norm_train_labels)
		norm_svr_pred = norm_svr_model.predict(norm_test_feats)
		unnorm_svr_pred = label_scaler.inverse_transform(norm_svr_pred)
		norm_svr_stat =("norm_svr_model", ) + validate(unnorm_svr_pred, true_labels, test_data[['race_id', 'finishing_position']])
		mess = "C "+str(c)+";epsilon "+str(eps)+str(norm_svr_stat)
		print(mess)
		log.append(mess)
with open("./svm1.log","w") as logf:
	logf.write("\n".join(log))

'''
svr_model = SVR(cache_size=2000,kernel='linear',C=1.0, epsilon=0.1)
svr_model.fit(train_features,train_labels)
svr_pred = svr_model.predict(test_features)
svr_stat = ("svr_model", ) + validate(svr_pred, true_labels, test_data[['race_id', 'finishing_position']])
print(svr_stat)

norm_svr_model = SVR(cache_size=2000,kernel='linear',C=1.0, epsilon=0.1)
norm_svr_model.fit(norm_train_feats,norm_train_labels)
norm_svr_pred = norm_svr_model.predict(norm_test_feats)
unnorm_svr_pred = label_scaler.inverse_transform(norm_svr_pred)
norm_svr_stat =("norm_svr_model", ) + validate(unnorm_svr_pred, true_labels, test_data[['race_id', 'finishing_position']])
print(norm_svr_stat)
'''

#4.1.2.
"""
gbrt_param_space = {"learning_rate":[0.05, 0.1,0.15,0.2],
				"n_estimators":[200,225,250,275,300],
				"max_depth":[2],
				"alpha":[0.9,0.2,0.5,1.5]
				}
for lr in gbrt_param_space["learning_rate"]:
	for ne in gbrt_param_space["n_estimators"]:
		for md in gbrt_param_space["max_depth"]:
			print("start training")
			gbrt_model = GradientBoostingRegressor(loss='quantile',alpha=0.9,verbose=0 ,learning_rate=lr, n_estimators=ne, max_depth=md)
			gbrt_model.fit(train_features,train_labels)
			gbrt_pred = gbrt_model.predict(test_features)
			print("finish training")
			gbrt_stat = ("gbrt_model", )+ validate(gbrt_pred, true_labels, test_data[['race_id','finishing_position']])
			print("lr "+str(lr)+";ne "+str(ne)+"; md "+str(md)+str(gbrt_stat))
#optimal seems: lr:0.1, ne:300, md:2

gbrt_model = GradientBoostingRegressor(loss='quantile',learning_rate=0.1, n_estimators=300, max_depth=2)
gbrt_model.fit(train_features,train_labels)
gbrt_pred = gbrt_model.predict(test_features)
gbrt_stat = ("gbrt_model", )+ validate(gbrt_pred, true_labels, test_data[['race_id','finishing_position']])
print(gbrt_stat)
print(gbrt_pred)

norm_gbrt_model = GradientBoostingRegressor(loss='quantile',learning_rate=0.1, n_estimators=300, max_depth=2)
norm_gbrt_model.fit(norm_train_feats,norm_train_labels)
norm_gbrt_pred = norm_gbrt_model.predict(norm_test_feats)
unnorm_gbrt_pred = label_scaler.inverse_transform(norm_gbrt_pred)
norm_gbrt_stat = ("norm_gbrt_model", )+ validate(unnorm_gbrt_pred, true_labels, test_data[['race_id','finishing_position']])
print(norm_gbrt_stat)
print(unnorm_gbrt_pred)
"""
