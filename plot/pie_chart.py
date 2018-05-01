import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fname = '../training.csv'
print('Reading dataframe from ', fname)
print('\n')
df = pd.read_csv(fname)
labels = []
values = []
#colours
wins = df[df.finishing_position == 1]
for i in range(df['draw'].astype('int').min(),df['draw'].astype('int').max()):
    values.append(len(wins[wins.draw == i]))
    labels.append(str(i))


fig = plt.figure(figsize=(6, 6))
plt.pie(values,labels=labels,autopct='%1.1f%%',radius=1.2,labeldistance=0.8)
plt.title('Fractions of number of winning of different draws')
plt.show()
#print(values)
#print(sum(values))
#print(labels)
