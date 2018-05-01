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


"""
We first choose the best regression model and classification model.
Our own strategy is that we use the regression model to predict the winner,
and then if the predicted winner is not in the HorseRankTop50Percent of the classification model.
We would reject the answer and save the bet.
"""
def combined_strat1(df,pred_rank,svr_pred):
    count = 0
    cost = 0
    revenue = 0
    Profit = 0
    rej = 0
    aggreg  = df.join(pd.DataFrame({'pred_time': svr_pred}))
    aggreg = aggreg.join(pred_rank)
    np.random.seed(0)
    for name, eachrace in aggreg.groupby('race_id'):
        eachrace = eachrace.sample(frac=1).reset_index(drop=True)
        Top1 = eachrace.nsmallest(1,'pred_time')
        Top1_rank = Top1['finishing_position'].iloc[0]
        Top1_WO = Top1['win_odds'].iloc[0]
        T_top50 = Top1['HorseRankTop50Percent'].iloc[0]
        if T_top50 != 0:
            cost += 1
            if Top1_rank == 1:
                revenue += Top1_WO
                count += 1
        else:
            rej += 1
    Profit = revenue - cost
    print("Cost: "+ str(cost))
    print("Revenue: "+ str(revenue))
    print("Profit: "+str(Profit))
    print("Correct Guess: "+ str(count))
    print("Reject: "+str(rej))
    return Profit

def combined_strat2(df,rf_pred,pred_rank,svr_pred):
    count = 0
    cost = 0
    revenue = 0
    Profit = 0
    rej = 0
    rf_pred.name = "rf_top50"
    pred_rank.name = "nb_top50"
    #rf_pred.columns = ['rf_top50']
    #pred_rank.columns = ['nb_top50']
    #print(rf_pred)
    aggreg  = df.join(pd.DataFrame({'pred_time': svr_pred}))
    aggreg = aggreg.join(rf_pred)
    aggreg = aggreg.join(pred_rank)
    np.random.seed(0)
    for name, eachrace in aggreg.groupby('race_id'):
        eachrace = eachrace.sample(frac=1).reset_index(drop=True)
        Top1 = eachrace.nsmallest(1,'pred_time')
        Top1_rank = Top1['finishing_position'].iloc[0]
        Top1_WO = Top1['win_odds'].iloc[0]
        rf_top50 = Top1['rf_top50'].iloc[0]
        nb_top50 = Top1['nb_top50'].iloc[0]
        if rf_top50 != 0 or nb_top50 !=0:
            cost += 1
            if Top1_rank == 1:
                revenue += Top1_WO
                count += 1
        else:
            rej += 1
    Profit = revenue - cost
    print("Cost: "+ str(cost))
    print("Revenue: "+ str(revenue))
    print("Profit: "+str(Profit))
    print("Correct Guess: "+ str(count))
    print("Reject: "+str(rej))
    return Profit

def combined_strat3(df,rf_pred,pred_rank,svr_pred):
    count = 0
    cost = 0
    revenue = 0
    Profit = 0
    rej = 0
    rf_pred.name = "rf_top50"
    pred_rank.name = "nb_top50"
    #rf_pred.columns = ['rf_top50']
    #pred_rank.columns = ['nb_top50']
    #print(rf_pred)
    aggreg  = df.join(pd.DataFrame({'pred_time': svr_pred}))
    aggreg = aggreg.join(rf_pred)
    aggreg = aggreg.join(pred_rank)
    np.random.seed(0)
    for name, eachrace in aggreg.groupby('race_id'):
        eachrace = eachrace.sample(frac=1).reset_index(drop=True)
        Top1 = eachrace.nsmallest(1,'pred_time')
        Top1_rank = Top1['finishing_position'].iloc[0]
        Top1_WO = Top1['win_odds'].iloc[0]
        rf_top50 = Top1['rf_top50'].iloc[0]
        nb_top50 = Top1['nb_top50'].iloc[0]
        if (rf_top50 != 0 or nb_top50 !=0) and Top1_WO > 9:
            cost += 1
            if Top1_rank == 1:
                revenue += Top1_WO
                count += 1
        else:
            rej += 1
    Profit = revenue - cost
    print("Cost: "+ str(cost))
    print("Revenue: "+ str(revenue))
    print("Profit: "+str(Profit))
    print("Correct Guess: "+ str(count))
    print("Reject: "+str(rej))
    return Profit


def winodds_strat(df, pred_time,pred_rank=None): # reject if win odds is low
    count = 0
    cost = 0
    revenue = 0
    Profit = 0
    rej = 0
    idx=0
    aggreg  = df.join(pd.DataFrame({'pred_time': pred_time}))
    np.random.seed(0)
    for name, eachrace in aggreg.groupby('race_id'):
        if idx < 10:
            #print(eachrace[['pred_time','win_odds','finishing_position']])
            idx += 1
        eachrace = eachrace.sample(frac=1).reset_index(drop=True) #avoid picking the first one if there is duplicated minimum time
        Top1 = eachrace.nsmallest(1,'pred_time') #choose the smallest prediction time
        Top1_rank = Top1['finishing_position'].iloc[0]
        Top1_WO = Top1['win_odds'].iloc[0]
        if Top1_WO > 9:
            cost += 1
            if int(Top1_rank) == 1:
            #print(revenue)
                revenue += Top1_WO
                count += 1
        else:
            rej += 1
    Profit = revenue - cost
    print("Cost: "+ str(cost))
    print("Revenue: "+ str(revenue))
    print("Profit: "+str(Profit))
    print("Correct Guess: "+ str(count))
    print("Reject: "+str(rej))
    return Profit

def default_strat(df, pred_time,pred_rank=None):
    count = 0
    cost = 0
    revenue = 0
    Profit = 0
    idx=0
    if pred_rank is not None:
        aggreg  = df.join(pred_rank)
        for name, eachrace in aggreg.groupby('race_id'):
            if idx < 10:
                print(eachrace[['HorseRankTop50Percent','RaceID','win_odds','finishing_position']])
                idx += 1
        return
    aggreg  = df.join(pd.DataFrame({'pred_time': pred_time}))
    np.random.seed(0)
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

#getting prediction
fname = './predictions/lr_predictions.csv'
print('Reading dataframe from ', fname)
print('\n')
lr_pred = pd.read_csv(fname)


fname = './predictions/nb_predictions.csv'
print('Reading dataframe from ', fname)
print('\n')
nb_pred = pd.read_csv(fname)


fname = './predictions/rf_predictions.csv'
print('Reading dataframe from ', fname)
print('\n')
rf_pred = pd.read_csv(fname)

fname = './predictions/lr_predictions.csv'
print('Reading dataframe from ', fname)
print('\n')
svm_pred = pd.read_csv(fname)
#print("LR Model")
#default_strat(test_data, None,pred_rank=lr_pred)
#print("NB Model")
#default_strat(test_data, None,pred_rank=nb_pred)
#print("RF Model")
#default_strat(test_data, None,pred_rank=rf_pred)
#print("SVM Model")
#default_strat(test_data, None,pred_rank=svm_pred)

gbrt_pred = gbrt_model.predict(test_features)
norm_gbrt_pred = norm_gbrt_model.predict(norm_test_feats)
svr_pred = svr_model.predict(test_features)
norm_svr_pred = norm_svr_model.predict(norm_test_feats)
#print(norm_svr_pred)

print("\nGBRT Model")
#default_strat(test_data, gbrt_pred,pred_rank=None)

print("\nNormalized GBRT Model")
#default_strat(test_data, norm_gbrt_pred,pred_rank=None)

print("\nSVR Model")
#default_strat(test_data, svr_pred,pred_rank=None)

print("\nNormalized SVR Model")
default_strat(test_data, norm_svr_pred,pred_rank=None)
print("\nWO strats")
winodds_strat(test_data, norm_svr_pred,pred_rank=None)
print("\nCombine Strat1")
combined_strat1(test_data , nb_pred['HorseRankTop50Percent'], norm_svr_pred)
print("\nCombine Strat2")
combined_strat2(test_data , rf_pred['HorseRankTop50Percent'], nb_pred['HorseRankTop50Percent'], norm_svr_pred)

print("\nCombine Strat3")
combined_strat3(test_data , rf_pred['HorseRankTop50Percent'], nb_pred['HorseRankTop50Percent'], norm_svr_pred)
