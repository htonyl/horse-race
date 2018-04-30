import pandas as pd
import numpy as np
import time

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from naive_bayes import NaiveBayes

## 3.0.0 Read train / test data
df_train = pd.read_csv('training.csv')
df_test = pd.read_csv('testing.csv')

def preprocess_classification(df):
    # Append number of candidates
    n_candid = df.groupby('race_id').size()
    n_candid.name = 'n_candid'
    df = df.join(n_candid, on='race_id', rsuffix='')
    
    # Transform finish time from str to float
    def to_second(v):
        l = list(map(int, v.split('.')))
        return l[0]*60+l[1]+l[2]*.001
    
    df.finish_time = df.finish_time.apply(to_second)
    
    # Calculate top1, top3, top_half
    top1 = (df.finishing_position == 1).astype(int)
    top1.name = 'top1'
    top3 = (df.finishing_position <= 3).astype(int)
    top3.name = 'top3'
    toph = (df.finishing_position <= df.n_candid/2).astype(int)
    toph.name = 'toph'
    df = pd.concat([df, top1, top3, toph], axis=1)
    return df

df_train = preprocess_classification(df_train)
df_test = preprocess_classification(df_test)

# Separate X, y
y_cols = [
    'finish_time',
    'finishing_position',
    'length_behind_winner',
    'running_position_1',
    'running_position_2',
    'running_position_3',
    'running_position_4',
    'running_position_5',
    'running_position_6',
    'top1',
    'top3',
    'toph'
    ]
X_train = df_train.drop(y_cols, 1)
y_train = df_train[y_cols]
X_test = df_test.drop(y_cols, 1)
y_test = df_test[y_cols]

# Evaluate models using the following metrics
y_metrics = ['top1', 'top3', 'toph']

X_train_ = X_train.select_dtypes(include=['number'])
X_test_ = X_test.select_dtypes(include=['number'])
print('Using features:')
for c in X_train_.columns:
    print('\t',c)
print('\n')

## 3.1-3 Fit & Predict
def conf_matrix(y, pred):
    cm = np.empty(4)
    cm[0] = np.logical_and(y==1,pred==1).sum()
    cm[1] = np.logical_and(y==1,pred==0).sum()
    cm[2] = np.logical_and(y==0,pred==1).sum()
    cm[3] = np.logical_and(y==0,pred==0).sum()
    return cm

def get_metrics(y, pred):
    cm = conf_matrix(y, pred)
    acc = (cm[0]+cm[3])/cm.sum()
    rec = cm[0]/(cm[0]+cm[1]) if cm[0]+cm[1] > 0 else 0
    prec = cm[0]/(cm[0]+cm[2]) if cm[0]+cm[2] > 0 else 0
    f1 = 1/(rec+prec) if rec+prec > 0 else float('inf') 
    return acc, rec, prec, f1

def fit_predict(model, X_cols=X_train_.columns):
    print('\nEvaluation: {}\n\taccuracy\tprecision\trecall\tf1\tfit\tscore\ttotal (s)'.format(model))
    for c in y_metrics:
        _t = time.time()
        model.fit(X_train_[X_cols], y_train[c])
        fit_t = time.time() - _t
        
        _t = time.time()
        pred = model.predict(X_test_[X_cols])
        pred_t = time.time() - _t

        acc, rec, prec, f1 = get_metrics(y_test[c], pred)
        print("{}:\t{:.9}%\t{:.6f}\t{:.5f}\t{:.5f}\t{:.4f}\t{:.4f}\t{:.4f}".format(c, acc*100, prec, rec, f1, fit_t, pred_t, fit_t+pred_t))
    pred = []
    return model, pred

kwargs = { 'random_state': 61052 }

## 3.1.1 Logistic Regression
print('## Logistic Regrssion ##')

model = LogisticRegression(**kwargs)
lr_model, pred = fit_predict(model)
print('\n')

## 3.1.2 Naive Bayes
print('## Naive Bayes ##')

model = GaussianNB(**kwargs)
nb_model, pred = fit_predict(model)

model = BernoulliNB(**kwargs)
nb_model, pred = fit_predict(model)
print('\n')

## 3.1.3 Support Vector Machine
print('## Support Vector Machine ##')

model = SVC(kernel='sigmoid', **kwargs)
svm_model, pred = fit_predict(model)
print('\n')

## 3.1.4 Random Forest
print('## Random Forest ##')

model = RandomForestClassifier(**kwargs)
rf_model, pred = fit_predict(model)
print('\n')

## 3.2 Save predictions
## 3.3 Evaluate predictions
## 3.4 Report

