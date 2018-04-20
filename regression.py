import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np



# Read
fname = './training.csv'
print('Reading dataframe from ', fname)
print('\n')
train_data = pd.read_csv(fname)

features = train_data.as_matrix(columns=['actual_weight', 'declared_horse_weight', 'draw','win_odds', 'jockey_ave_rank', 'trainer_ave_rank', 'recent_ave_rank', 'race_distance'])

labels = train_data.as_matrix(columns=['finish_time']) 
for i in range(len(labels)):
	min,sec,msec = labels[i][0].split('.')
	labels[i][0] = float(min)*60*60 + float(sec) * 60 + float(msec)
labels = np.reshape(labels,-1)
#print(features[0])
#print(labels)

#4.1.1.
svr_model = SVR()
svr_model.fit(features,labels)


#4.1.2.
gbrt_model = GradientBoostingRegressor()
gbrt_model.fit(features,labels)






