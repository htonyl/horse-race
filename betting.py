from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def extract(df):
	features = df.as_matrix(columns=['actual_weight', 'declared_horse_weight', 'draw','win_odds', 'jockey_ave_rank', 'trainer_ave_rank', 'recent_ave_rank', 'race_distance'])
	labels = df.as_matrix(columns=['finish_time'])
	for i in range(len(labels)):
		min,sec,msec = labels[i][0].split('.')
		labels[i][0] = float(min)*60 + float(sec) + float(msec)/100
	labels = np.reshape(labels,-1)
	return features, labels

def default_strat(df, pred_time,pred_rank=None):
    aggreg  = df.join(pd.DataFrame({'pred_time': pred_time}))
    np.random.seed(0)
    count = 0
    cost = 0
    revenue = 0
    Profit = 0
    idx=0
    for name, eachrace in aggreg.groupby('race_id'):
        if idx < 10:
            #print(eachrace[['pred_time','win_odds','finishing_position']])
            idx += 1
        cost += 1
        eachrace = eachrace.sample(frac=1).reset_index(drop=True) #avoid picking the first one if there is duplicated minimum time
        Top1 = eachrace.nsmallest(1,'pred_time') #choose the smallest prediction time
        Top1_rank = Top1['finishing_position'].iloc[0]
        Top1_WO = Top1['win_odds'].iloc[0]
        if int(Top1_rank) == 1:
            #print(revenue)
            revenue += Top1_WO
            count += 1
    Profit = revenue - cost
    print("Cost: "+ str(cost))
    print("Revenue: "+ str(revenue))
    print("Profit: "+str(Profit))
    print("Correct Guess: "+ str(count))
    return Profit

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
norm_gbrt_model = joblib.load('./model/norm_GBRT.pkl')
gbrt_model = joblib.load('./model/GBRT.pkl')
norm_svr_model = joblib.load('./model/norm_SVR.pkl')
svr_model = joblib.load('./model/SVR.pkl')

feat_scaler = StandardScaler()
'''
because normalized model trained with normalized training data
i use mean and variance of training data to normalize testing data
'''
norm_train_feats = feat_scaler.fit_transform(train_features)
norm_test_feats = feat_scaler.transform(test_features)

gbrt_pred = gbrt_model.predict(test_features)
norm_gbrt_pred = norm_gbrt_model.predict(norm_test_feats)
svr_pred = svr_model.predict(test_features)
norm_svr_pred = norm_svr_model.predict(norm_test_feats)
#print(norm_svr_pred)

print("GBRT Model")
default_strat(test_data, gbrt_pred,pred_rank=None)

print("Normalized GBRT Model")
default_strat(test_data, norm_gbrt_pred,pred_rank=None)

print("SVR Model")
default_strat(test_data, svr_pred,pred_rank=None)

print("Normalized SVR Model")
default_strat(test_data, norm_svr_pred,pred_rank=None)
