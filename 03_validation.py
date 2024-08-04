# %%
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error




# %%
valid_df = pd.read_csv('./processed_data/validation_set.csv')
valid_df['date'] = pd.to_datetime(valid_df['date'])


with open('./model/magic_number.pkl', 'rb') as f: magic_number = pickle.load(f)
with open('./model/best_model.pkl', 'rb') as f: model = pickle.load(f)



# %%
valid_df['flow_pred'] = valid_df.apply(
    # lambda row: model.predict(row.drop(['date', 'flow']).to_frame().T)[0] - magic_number
    lambda row: model.predict(row.drop(['date', 'flow']).to_frame().T)[0] - magic_number, axis=1
)




# %%
fig, ax = plt.subplots()
valid_df[['flow', 'flow_pred']].plot(ax=ax)
fig.savefig('./results/validation/validatiion_flow_vs_flow_pred.jpg', bbox_inches='tight')


# %%
fig, ax = plt.subplots()
(valid_df['flow'] -valid_df['flow_pred']).plot()
fig.savefig('./results/validation/validatiion_mae.jpg', bbox_inches='tight')


# %%
mape_corrected = mean_absolute_percentage_error(valid_df['flow'] + magic_number, valid_df['flow_pred'] + magic_number) 
mape = mean_absolute_percentage_error(valid_df['flow'], valid_df['flow_pred']) 
mae =  mean_absolute_error(valid_df['flow'], valid_df['flow_pred'])

print('MAPE', mape)
print('MAPE [corrected]', mape_corrected)
print('MAE', mae)

f = open('./results/validation/validatiion_summary.txt', 'w')
f.write(f'MAPE: {mape_corrected}')
f.write(f'MAPE [corrected]: {mape}')
f.write(f'MAE [corrected]: {mae}')
f.close()



