# %%
import mysql.connector
import pandas as pd
import numpy as np

pd.options.display.max_columns = 999

# %%
def column_to_lowercase(df):
    for col in df.columns: 
        df = df.rename(columns={col:col.replace(' ', '_').lower()})
    return df


def fill_with_median(df, column_name):
    for i in df.index:
        if pd.isna(df.loc[i, column_name]):
            prev = df.loc[:i, column_name].dropna().last_valid_index()
            next = df.loc[i:, column_name].dropna().first_valid_index()
            if pd.notna(prev) and pd.notna(next):
                df.loc[i, column_name] = np.median([df.loc[prev, column_name], df.loc[next, column_name]])
    


# %% [markdown]
# # load data

# %%
# Establish a connection to the MySQL database
connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="pass1234",
    database="sys"
)

cursor = connection.cursor()

# query to merge trips and stations
query = """
    SELECT 
        trips.*, 
        start_stations.id AS start_id, 
        start_stations.name AS start_name, 
        start_stations.lat AS start_lat, 
        start_stations.long AS start_lon,
        start_stations.zip as start_zip,
        start_stations.dock_count AS start_dock_count,
        
        end_stations.id AS end_id, 
        end_stations.name AS end_name, 
        end_stations.lat AS end_lat, 
        end_stations.long AS end_lon,
        end_stations.zip as end_zip,
        end_stations.dock_count AS end_dock_count


    FROM 
        trips 
    JOIN 
        stations AS start_stations 
    ON 
        trips.start_station = start_stations.id 
    JOIN 
        stations AS end_stations 
    ON 
        trips.end_station = end_stations.id;
"""
trips_x_stations = pd.read_sql(query, connection)

# column to lowercase
trips_x_stations = column_to_lowercase(trips_x_stations)

# cast to datetime the column
trips_x_stations['start_date'] = pd.to_datetime(trips_x_stations['start_date'], dayfirst=True)
trips_x_stations['end_date'] = pd.to_datetime(trips_x_stations['end_date'], dayfirst=True)
cursor.close()

print(trips_x_stations.shape)
trips_x_stations.head()

# %%
# load and cast weather
weather = pd.read_csv('./data/weather_data.csv')
weather = column_to_lowercase(weather)
weather['date'] = pd.to_datetime(weather['date'], dayfirst=True)
print(weather.shape)
weather.head()

# %% [markdown]
# # manage missing data in weateher

# %%
weather = weather.sort_values(by='date')

# by intution, filled with sunny
weather['events'] = weather['events'].fillna('sunny')

# compute the columns with nans
columns_with_nans = [c for c in weather.columns if weather[weather[c].isna()].shape[0] > 0  ]

# creaete a dict for each zip
zip_dict = {zzip: weather[weather['zip'] == zzip] for zzip in weather.zip.unique()}

# for each zip, column with nan fill the nans with median between the two closest value to the nan
# assumption: weather data could not vary so widley in a limted amount of time
for key in zip_dict.keys():
    for col_with_nans in columns_with_nans:
        fill_with_median(zip_dict[key], col_with_nans)
weather = pd.concat(zip_dict.values())

# some columns are not properlu filled up, so they are removed
col_still_with_nans = [col for col in weather.columns if weather[weather[col].isna()].shape[0] > 0  ]
weather = weather.drop(col_still_with_nans, axis=1)

for col in weather.columns:
    assert weather[weather[col].isna()].shape[0] == 0


# %%
# to merge without asof, i extract the day date and cast int a string
trips_x_stations['start_date_day'] = trips_x_stations['start_date'].dt.date.astype(str)
trips_x_stations['end_date_day'] = trips_x_stations['end_date'].dt.date.astype(str)
weather['date_merge'] = weather['date'].dt.date.astype(str)

# %% [markdown]
# # Transform the trips into flows

# %%
# stringfication of date hour
trips_x_stations['date_hour'] = trips_x_stations.start_date.dt.date.astype(str) + ' ' + trips_x_stations.start_date.dt.hour.astype(str) +':00:00'
# counting all trips leaving at each hour for each statons
start_per_date_per_loc = trips_x_stations.groupby(['date_hour' ,'start_station', 'start_zip']).count().trip_id.reset_index()
# coliumn standardization
start_per_date_per_loc = start_per_date_per_loc.rename(columns = {'start_station':'station', 'trip_id':'trip_in', 'start_zip':'zip'})


# stame procedure for flow out
trips_x_stations['date_hour'] = trips_x_stations.start_date.dt.date.astype(str) + ' ' + trips_x_stations.start_date.dt.hour.astype(str) +':00:00'
end_per_date_per_loc   = trips_x_stations.groupby(['date_hour', 'end_station', 'end_zip']).count().trip_id.reset_index()
end_per_date_per_loc = end_per_date_per_loc.rename(columns = {'end_station':'station', 'trip_id':'trip_out', 'end_zip':'zip'})

# merge #bikes leaving and #bikes starting
flow_per_stat = start_per_date_per_loc.merge(
    end_per_date_per_loc,
    on=['date_hour', 'station', 'zip'],
)
# flow computaation
flow_per_stat['flow'] = flow_per_stat['trip_in'] - flow_per_stat['trip_out']

# date casting
flow_per_stat['date'] = pd.to_datetime(flow_per_stat['date_hour'])

# date stringication for further merges
flow_per_stat['date_merge'] = flow_per_stat['date'].dt.date.astype(str)
flow_per_stat = flow_per_stat.drop(['date_hour', 'trip_in', 'trip_out'], axis=1)

# %%
# # merge the weather features
flow_per_stat
df = flow_per_stat.merge(
    weather.drop('date', axis=1),
    on = ['date_merge', 'zip'],
)
df = df.drop(['date_merge'], axis=1)
df['date'] = pd.to_datetime(df['date'])

# # final_df
print('final shape', df.shape)

# %%
df.head()

# %% [markdown]
# # adding new features

# %%
# is weekend
df['is_weekend'] = df.date.dt.day_of_week >= 5
df['is_workingday'] = df.date.dt.day_of_week < 5

# one hot encodinf of string colums
df = pd.get_dummies(df, columns=['events'])

# # #compute flow lags per pount

def calculate_previous_flow(df, delta_hours=1):
    # Assicurati che 'date' sia di tipo datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Imposta 'station' e 'date' come multi-indice
    df = df.set_index(['station', 'date'])
    
    # Ordina l'indice per assicurare che il shift operi correttamente
    df.sort_index(inplace=True)
    
    # Applica shift dentro ogni gruppo di stazione
    df[f'last_{delta_hours}_flow'] = df.groupby(level='station')['flow'].shift(delta_hours)
    
    # Reset dell'indice
    df = df.reset_index()
    
    return df


df = calculate_previous_flow(df, 1)
df = calculate_previous_flow(df, 2)
df = calculate_previous_flow(df, 4)
df = calculate_previous_flow(df, 5)



df['hour'] = df['date'].dt.hour
df['dow'] = df['date'].dt.day_of_week
df['month'] = df['date'].dt.month
df['woy'] = df['date'].dt.isocalendar().week





# %%
validation_set = df[
    (df.date >= pd.to_datetime('2015-08-01')) &
    (df.date < pd.to_datetime('2015-09-01'))
]
validation_set.to_csv('./processed_data/validation_set.csv', index=False)

train_and_test = df[(df.date < pd.to_datetime('2015-08-01'))]
train_and_test.to_csv('./processed_data/train_and_test.csv', index=False)

# %%
# grp = df.groupby([df.start_date.dt.date, 'start_id']).count().trip_id.reset_index()

# for stat in grp.start_id:
#     grp[grp == stat].trip_id.plot()

# %%



