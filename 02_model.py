# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from modules.mc_forrest import mc_forrest
from modules.mc_xgb import mc_xgb

import pickle

# %%
df = pd.read_csv('./processed_data/train_and_test.csv')
df['date'] = pd.to_datetime(df['date'])


X = df.drop(['date', 'flow'], axis=1)
y = df['flow']

X.head(10)

# %%
y.head()

# %%
# Dividi il dataset in train e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
magic_number = abs(y_train.min())

y_train = y_train + magic_number
y_test = y_test + magic_number

res = dict(
    RandomForrest = mc_forrest(X_train, y_train, X_test, y_test),
    XGB = mc_xgb(X_train, y_train, X_test, y_test),
)

# %%
tab = pd.DataFrame(res).T[['mape', 'mae']]
fig, ax = plt.subplots()
tab['mape'].plot.bar(color='blue', ax=ax, title='mape')
fig.savefig('./results/train/train_mape.jpg', bbox_inches='tight')

fig, ax = plt.subplots()
tab['mae'].plot.bar(color='orange', ax=ax, title='mae')
fig.savefig('./results/train/train_mae.jpg', bbox_inches='tight')



# %%
tab['mae'].idxmin()

# %%
best_model = res[tab['mae'].idxmin()]['model']

with open('./model/best_model.pkl', 'wb') as f: pickle.dump(best_model, f)
with open('./model/magic_number.pkl', 'wb') as f: pickle.dump(magic_number, f)

f = open('./results/train/summary.txt', 'w')
f.write(f'model:{best_model}')
f.write(f'magic_number:{magic_number}')
f.close()
tab.to_csv('./results/train/stats.csv',)



