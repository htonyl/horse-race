import pandas as pd
import numpy as np
import time, os, pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

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
y_metrics = ['toph']

X_train_ = X_train.filter(items=['recent_ave_rank', 'jockey_ave_rank'])
X_test_ = X_test.filter(items=['recent_ave_rank', 'jockey_ave_rank'])
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
    # Validation split
    X_t, X_v, y_t, y_v = train_test_split(X_train_, y_train, test_size=0.3, **kwargs)
    pred_save = np.empty((len(X_test_), 3))
    for idx, c in enumerate(y_metrics):
        _t = time.time()
        model.fit(X_t[X_cols], y_t[c])
        fit_t = time.time() - _t
        
        # Validation
        _t = time.time()
        pred = model.predict(X_v[X_cols])
        pred_t = time.time() - _t

        acc, rec, prec, f1 = get_metrics(y_v[c], pred)
        print("{}_V:\t{:.9}%\t{:.6f}\t{:.5f}\t{:.5f}\t{:.4f}\t{:.4f}\t{:.4f}".format(c, acc*100, prec, rec, f1, fit_t, pred_t, fit_t+pred_t))

        # Test
        _t = time.time()
        pred = model.predict(X_test_[X_cols])
        pred_t = time.time() - _t

        acc, rec, prec, f1 = get_metrics(y_test[c], pred)
        pred_save[:, idx] = pred
        print("{}_T:\t{:.9}%\t{:.6f}\t{:.5f}\t{:.5f}\t{:.4f}\t{:.4f}\t{:.4f}".format(c, acc*100, prec, rec, f1, fit_t, pred_t, fit_t+pred_t))
    return model, pred_save


kwargs = { 'random_state': 61052 }
pred = {}

print('## Support Vector Machine ##')

fname = 'svm_model.pkl'
if os.path.isfile(fname):
    print('Loading model from {}....'.format(fname))
    with open(fname, 'rb') as f:
        svm_model = pickle.load(f)
else:
    model = SVC(kernel='linear', **kwargs)
    svm_model, pred['svm_sig'] = fit_predict(model)
    with open(fname, 'wb') as f:
        pickle.dump(svm_model, f)
print('\n')

print('Plot decision boundary of SVM')

def make_meshgrid(x, y, h=.02):
    print('  preparing meshgrid...')
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    print('  plotting contours')
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

X0, X1 = X_train_.recent_ave_rank, X_train_.jockey_ave_rank 
xx, yy = make_meshgrid(X0, X1)
fig, ax = plt.subplots(1,1)
plot_contours(ax, svm_model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y_train.toph, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xlabel('Recent Ave Rank')
ax.set_ylabel('Jockey Ave Rank')
ax.set_title('Decision Boundary of SVM (linear kernel)')
