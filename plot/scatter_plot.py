import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fname = '../data/race-result-horse.csv'
print('Reading dataframe from ', fname)
print('\n')

df = pd.read_csv(fname)
cond = np.where([not (type(s) == type('') and s.isdigit()) for s in df['finishing_position']])
df = df.drop(cond[0], axis=0)

horses = df.groupby('horse_name')
h_numwin = horses.apply(lambda v: len(v[v.finishing_position == "1"]))
h_numwin.name = 'number_of_win'
h_winrate = horses.apply(lambda v: float(len(v[v.finishing_position == "1"]))/len(v))
h_winrate.name = 'win_rate'

jocks = df.groupby('jockey')
j_numwin = jocks.apply(lambda v: len(v[v.finishing_position == "1"]))
j_numwin.name = 'number_of_win'
j_winrate = jocks.apply(lambda v: float(len(v[v.finishing_position == "1"]))/len(v))
j_winrate.name = 'win_rate'


fig, ax = plt.subplots(figsize=(10, 5))
plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
ax.scatter(h_winrate,h_numwin,s=10)
for i in range(len(h_winrate)):
    if h_numwin.iloc[i] > 3 and h_winrate.iloc[i] > 0.5 :
        ax.annotate(str(h_numwin.axes[0][i]) ,(h_winrate.iloc[i]+0.01,h_numwin.iloc[i]))
        #print(df[df.horse_name == str(h_numwin.axes[0][i])])
plt.xlabel("Win Rate")
plt.ylabel("Number of wins")
plt.title("Winning statistics of horses")
plt.show()

fig, ax = plt.subplots()
ax.scatter(j_winrate,j_numwin,s=10)
for i in range(len(j_winrate)):
    if j_numwin.iloc[i] > 100 and j_winrate.iloc[i] > 0.2 :
        ax.annotate(str(j_numwin.axes[0][i]) ,(j_winrate.iloc[i]+0.005,j_numwin.iloc[i]))
        #print(df[df.jockey == str(j_numwin.axes[0][i])])
plt.xlabel("Win Rate")
plt.ylabel("Number of wins")
plt.title("Winning statistics of jockeys")
plt.show()
