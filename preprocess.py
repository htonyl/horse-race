import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def print_stat(df):
    print('Read {} rows {} columns, with column names: '.format(df.shape[0], df.shape[1]))
    for c in df.columns:
        print(c)
    print('\n')
    
    print('Simple statistics of numerical columns: \n', df.describe().T)
    print('\n')

# 2.2.1 Read dataframe
fname = 'data/race-result-horse.csv'
print('Reading dataframe from ', fname)
print('\n')

df = pd.read_csv(fname)
print_stat(df)

# Drop rows with finishing_position is not a number (special run)
cond = np.where([not (type(s) == type('') and s.isdigit()) for s in df['finishing_position']])
df = df.drop(cond[0], axis=0)

# Cast finishing_position to int
df.finishing_position = df.finishing_position.apply(pd.to_numeric)

## 2.2.2 Recent performance of each horse
groupby_hid = df.groupby('horse_id')

# Append recent_6_runs
print('Appending column: recent_6_runs')
recent_6 = groupby_hid['finishing_position'].apply(lambda v: "/".join([str(int(n)) for n in v[-6:]]))
recent_6.name = 'recent_6_runs'
df = df.join(recent_6, on='horse_id', rsuffix='')

# Append recent_ave_rank
print('Appending column: recent_ave_rank')
recent_ave = groupby_hid['finishing_position'].apply(lambda v: np.mean(v[-6:]))
recent_ave.name = 'recent_ave_rank'
df = df.join(recent_ave, on='horse_id', rsuffix='')

print('\n')

## 2.2.3 Index horse, jockey, trainer
hid_map = {}
for i, j in enumerate(df.horse_id.unique()):
    hid_map[j] = i
print('Index horse: {} unique horses in total'.format(len(hid_map.keys())))

jck_map = {}
for i, j in enumerate(df.jockey.unique()):
    jck_map[j] = i
print('Index jockey: {} unique jockies in total'.format(len(jck_map.keys())))

trn_map = {}
for i, j in enumerate(df.trainer.unique()):
    trn_map[j] = i
print('Index trainer: {} unique trainers in total'.format(len(trn_map.keys())))

print('\n')

## 2.2.4 Distance of race
fname = 'data/race-result-race.csv'
print('Reading dataframe from ', fname)
print('\n')

df_r = pd.read_csv(fname)

print_stat(df_r)

# Append race_distance
print('Appending column: race_distance')
race_series = pd.Series(data=list(df_r.race_distance), index=df_r.race_id)
race_series.name = 'race_distance' 
df = df.join(race_series, on='race_id', rsuffix='')

print('\n')

## 2.2.5 Test / Train split

# Split train and test
split_pt = '2016-327'
X_train = df[df.race_id <= split_pt]
X_test = df[df.race_id > split_pt]
print('Split dataframe into # of train / test: {} / {}'.format(len(X_train), len(X_test)))

# Append jockey_ave_rank
print('Appending column: jockey_ave_rank')
jck_ave = X_train.groupby('jockey')['finishing_position'].mean()
jck_ave.name = 'jockey_ave_rank'
X_train = X_train.join(jck_ave, on='jockey', rsuffix='')
X_test = X_test.join(jck_ave, on='jockey', rsuffix='')
X_test.jockey_ave_rank = X_test.jockey_ave_rank.replace(float('nan'), 7)

# Append trainer_ave_rank
print('Appending column: trainer_ave_rank')
trn_ave = X_train.groupby('trainer')['finishing_position'].mean()
trn_ave.name = 'trainer_ave_rank'
X_train = X_train.join(trn_ave, on='trainer', rsuffix='')
X_test = X_test.join(trn_ave, on='trainer', rsuffix='')
X_test.trainer_ave_rank = X_test.trainer_ave_rank.replace(float('nan'), 7)

# Save to csv
X_train.to_csv('training.csv')
X_test.to_csv('testing.csv')
print('Saved preprocessed data to training.csv / testing.csv')
