import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fname = '../training.csv'
print('Reading dataframe from ', fname)
print('\n')
df = pd.read_csv(fname)

while True:
    horseid = input("Enter horse id: ")
    if horseid == "exit":
        break
    recent_6 = df[df.horse_id == str(horseid)][-6:]
    recent_6.name = 'recent_6_runs'
    if recent_6.empty:
        print("no record")
        continue
    rank=[int(x) for x in recent_6.as_matrix(columns=['finishing_position'])]
    gameid=recent_6.as_matrix(columns=['race_id'])
    plt.plot(gameid,rank)
    plt.xlabel("Game ID")
    plt.ylabel("Rank")
    plt.title("Recent 6 races of horse "+horseid)
    plt.show()
    #print(rank)
    #print(gameid)
