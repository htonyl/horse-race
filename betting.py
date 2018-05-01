from sklearn.externals import joblib
import pandas as pd
import numpy as np

def extract(df):
	features = df.as_matrix(columns=['actual_weight', 'declared_horse_weight', 'draw','win_odds', 'jockey_ave_rank', 'trainer_ave_rank', 'recent_ave_rank', 'race_distance'])
	labels = df.as_matrix(columns=['finish_time'])
	for i in range(len(labels)):
		min,sec,msec = labels[i][0].split('.')
		labels[i][0] = float(min)*60 + float(sec) + float(msec)/100
	labels = np.reshape(labels,-1)
	return features, labels

gbrt_model = joblib.load('./model/GBRT.pkl')

fname = './testing.csv'
print('Reading dataframe from ', fname)
print('\n')
test_data = pd.read_csv(fname)

test_features , true_labels = extract(test_data)

gbrt_pred = gbrt_model.predict(test_features)

np.random.seed(0)

aggreg  = test_data.join(pd.DataFrame({'pred_time': gbrt_pred}))


cost = 0
revenue = 0
for name, eachrace in aggreg.groupby('race_id'):
    cost += 1
    eachrace = eachrace.sample(frac=1).reset_index(drop=True)
    Top1 = eachrace.nsmallest(1,'pred_time')
    Top1_rank = Top1['finishing_position'].iloc[0]
    Top1_WO = Top1['win_odds'].iloc[0]
    if int(Top1_rank) == 1:
        revenue += Top1_WO
print("Cost: "+ str(cost))
print("Revenue: "+ str(revenue))
print("Profit: "+str(revenue - cost))
