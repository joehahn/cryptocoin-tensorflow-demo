#predict_crypto_price.py
#
#by Joe Hahn
#jmh.datasciences@gmail.com
#21 January 2018

#this demo was adapted from this blog,  
#https://dashee87.github.io/deep%20learning/python/predicting-cryptocurrency-prices-with-deep-learning
#and notebook
#https://github.com/dashee87/blogScripts/blob/master/Jupyter/2017-11-20-predicting-cryptocurrency-prices-with-deep-learning.ipynb

#import matplotlib pandas etc
import matplotlib
#uncomment the following when on headless EC2 instance
matplotlib.use('agg') 
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd
from matplotlib import rcParams
sns.set(font_scale=1.5, font='DejaVu Sans')
pd.set_option('display.max_columns', None)

#get starting time
import time
time_start = time.time()

#read bitcoin market data over selected dates
start_date = '20160501'
stop_date = '20180113'
url = "https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=" + \
    start_date + "&end=" + stop_date
print url
df = pd.read_html(url)[0]
#convert string dates to datetime
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
#daily fractional price change
df['daily_change'] = (df['Close'] - df['Open'])/df['Open']
#subsequent fractional price change
df['next_daily_change'] = df['daily_change'].shift(-1)
#close off high
df['close_off_high'] = 2*(df['High']- df['Close'])/(df['High']-df['Low'])-1
#volatility
df['volatility'] = (df['High'] - df['Low'])/df['Open']
#modify column names
cols = [col + '_bit' for col in df.columns]
cols[0] = 'Date'
df.columns = cols
bitcoin = df
print bitcoin['Date'].dt.date.min(), bitcoin['Date'].dt.date.max()
print bitcoin.dtypes
print bitcoin.shape
bitcoin.tail()

#read etherium market data over selected dates
url = "https://coinmarketcap.com/currencies/ethereum/historical-data/?start=" + \
    start_date + "&end=" + stop_date
print url
df = pd.read_html(url)[0]
#convert string dates to datetime
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
#daily fractional price change
df['daily_change'] = (df['Close'] - df['Open'])/df['Open']
#subsequent fractional price change
df['next_daily_change'] = df['daily_change'].shift(-1)
#close off high
df['close_off_high'] = 2*(df['High']- df['Close'])/(df['High']-df['Low'])-1
#volatility
df['volatility'] = (df['High'] - df['Low'])/df['Open']
#modify column names
cols = [col + '_ether' for col in df.columns]
cols[0] = 'Date'
df.columns = cols
ethereum = df
print ethereum['Date'].dt.date.min(), ethereum['Date'].dt.date.max()
print ethereum.dtypes
print ethereum.shape
ethereum.head()

#merge bitcoin and ethereum data
coins = bitcoin.merge(ethereum, on='Date', how='inner')
print coins.shape
print coins.dtypes
coins.head()

#plot closing prices vs time
fig, ax = plt.subplots(figsize=(15, 6))
xp = coins['Date']
yp = coins['Close_bit']/1000.0
p = ax.plot(xp, yp, linestyle='-', label='bitcoin')
xp = coins['Date']
yp = coins['Close_ether']/1000.0
ax.plot(xp, yp, linestyle='-', label='ethereum')
ax.set_xlabel('date')
ax.set_ylabel('closing price    (K$)')
ax.set_yscale('log')
ax.set_ylim(0.001, 100)
ax.legend()
plt.savefig('figs/price.png')

#plot volumes vs time
fig, ax = plt.subplots(figsize=(15, 6))
xp = coins['Date']
yp = coins['Volume_bit']/1.0e9
ax.plot(xp, yp, linestyle='-', label='bitcoin')
xp = coins['Date']
yp = coins['Volume_ether']/1.0e9
ax.plot(xp, yp, linestyle='-', label='ethereum')
ax.set_xlabel('date')
ax.set_ylabel('volume    ()')
ax.set_yscale('log')
ax.set_ylim(1.0e-3, 100)
ax.legend()
plt.savefig('figs/volume.png')

#model will be trained on data collected early 2017, and tested against subsequent data
train_start_date = '2017-02-28'
test_start_date = '2017-11-15'
df = coins
train_idx = (df['Date'] >= train_start_date) & (df['Date'] < test_start_date)
fig, ax = plt.subplots(figsize=(15, 6))
xp = df[train_idx]['Date']
yp = df[train_idx]['Close_ether']/1000.0
ax.plot(xp, yp, linestyle='-', label='ethereum training data')
test_idx = (df['Date'] >= test_start_date)
xp = df[test_idx]['Date']
yp = df[test_idx]['Close_ether']/1000.0
ax.plot(xp, yp, linestyle='-', label='ethereum testing data')
idx = coins['Date'] >= train_start_date
xp = coins[idx]['Date']
yp = coins[idx]['Close_bit']/1.0e4
ax.plot(xp, yp, linestyle='-.', label='bitcoin (10K$)', alpha=0.7)
ax.set_xlabel('date')
ax.set_ylabel('closing price    (K$)')
ax.set_title('ethereum')
ax.legend()
plt.savefig('figs/training.png')

#select index, feature and target columns from records in coins having desired Dates
index_col = 'Date'
feature_cols = ['Close_bit', 'Volume_bit', 'close_off_high_bit', 'volatility_bit',
    'daily_change_bit', 'Close_ether', 'Volume_ether', 'close_off_high_ether',
    'volatility_ether', 'daily_change_ether']
target_col = 'next_daily_change_ether'
xy = coins[feature_cols + [target_col]]
index = coins[index_col]
x = xy[feature_cols]
y = xy[target_col]
print index.shape, x.shape, y.shape, xy.shape
print index.dt.date.min(), index.dt.date.max()
x.tail()

#standardize the x-columns
from sklearn import preprocessing
train_idx = (index >= train_start_date) & (index < test_start_date)
scaler = preprocessing.StandardScaler().fit(x[train_idx])
x_scaled = pd.DataFrame(scaler.transform(x), columns=feature_cols, index=y.index)
x_scaled.tail()

#get number of available timesteps, number of lagged timesteps, and number of data features
N_timesteps = x_scaled.shape[0]
N_lagged_steps = 4
N_features = x_scaled.shape[1]
print 'N_timesteps = ', N_timesteps
print 'N_lagged_steps = ', N_lagged_steps
print 'N_features = ', N_features

#restack 2D x_scaled array as 3D array = stack of 2D lagged features
x_list = []
for idx, row in x_scaled.iterrows():
    x_sub_df = x_scaled.loc[idx-N_lagged_steps+1:idx]
    x_sub_array = x_sub_df.values
    if (len(x_sub_array) < N_lagged_steps):
        x_sub_array = np.zeros((N_lagged_steps, N_features))
    x_list.append(x_sub_array)
x_array = np.array(x_list)
x_array.shape

#split x,y into training and testing numpy arrays spanning the desired dates
train_idx = (index >= train_start_date) & (index < test_start_date)
N_timesteps = train_idx.sum()
print 'N_timesteps = ', N_timesteps
x_train = x_array[train_idx]
y_train = y[train_idx].values
print 'x_train.shape = ', x_train.shape
print 'y_train.shape = ', y_train.shape
test_idx = (index >= test_start_date)
x_test = x_array[test_idx]
y_test = y[test_idx].values
print 'x_test.shape = ', x_test.shape
print 'y_test.shape = ', y_test.shape
print 'N_timesteps = ', N_timesteps
print 'N_lagged_steps = ', N_lagged_steps
print 'N_features = ', N_features
#the validation dataset will be a 30% random sample of test records
valid_idx = test_idx[test_idx == True].sample(frac=0.3).index
x_valid = x_array[valid_idx]
y_valid = y[valid_idx].values
print 'x_valid.shape = ', x_valid.shape
print 'y_valid.shape = ', y_valid.shape

#import keras modules
from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM, Dropout

#keras docs claim that it will use laptop's GPU by default, but keras appears to be using laptop's CPU :( 
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

#this helper function assembles the LSTM model
def lstm_model(N_neurons, input_shape, output_size, activ_func='linear', dropout=0.25, loss='mae', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(N_neurons, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    return model

#train LSTM model...1 minute on macbook pro
N_neurons = 12
input_shape = x_train[0].shape
output_size = 1
N_epochs = 21
np.random.seed(124)
model = lstm_model(N_neurons, input_shape, output_size, activ_func='tanh', dropout=0.35)
loss_history = model.fit(x_train, y_train, epochs=N_epochs, batch_size=1, validation_data=(x_valid, y_valid),
    verbose=1, shuffle=True)
print ' training  loss = ', loss_history.history['loss'][-1]
print 'validation loss = ', loss_history.history['val_loss'][-1]

#plot loss function vs training epoch
fig, ax = plt.subplots(1,1, figsize=(15, 6))
ax.plot(loss_history.epoch, loss_history.history['loss'], label='training loss')
ax.plot(loss_history.epoch, loss_history.history['val_loss'], label='validation loss')
ax.set_title('loss function vs training epoch')
ax.set_ylabel('Mean Absolute Error (MAE)')
ax.set_xlabel('training epoch')
ax.legend()
plt.savefig('figs/loss.png')

#generate predicted y_test = ethereum's fractional next-day gain
col = target_col + '_pred'
y_test_pred = pd.DataFrame(model.predict(x_test), index=index[test_idx], columns=[col])
print x_test.shape, y_test.shape
y_test_pred.head()

#flatten 3D x_train and x_test arrays > 2D array
x_list = []
for timestamp_idx in range(len(x_train)):
    x_list += [x_train[timestamp_idx].flatten()]
x_train_2D = np.array(x_list)
print x_train_2D.shape, y_train.shape
x_list = []
for timestamp_idx in range(len(x_test)):
    x_list += [x_test[timestamp_idx].flatten()]
x_test_2D = np.array(x_list)
print x_test_2D.shape, y_test.shape

#fit linear regression model to training data
from sklearn import linear_model
LR_model = linear_model.LinearRegression()
LR_model.fit(x_train_2D, y_train)
y_test_pred_LR = LR_model.predict(x_test_2D)

#plot actual and predicted vs date
sns.set(font_scale=1.4)
fig, ax = plt.subplots(figsize=(15, 6))
xp = y_test_pred.index
yp = y_test_pred
ax.plot(xp, yp, marker='o', markersize=5, linestyle='-', label='LSTM prediction', alpha=0.7)
yp = y_test
ax.plot(xp, yp, marker='o', markersize=5, linestyle='-', label='actual', alpha=0.8)
yp = y_test_pred_LR
ax.plot(xp, yp, marker='o', markersize=4, linestyle='-.', label='LR prediction', alpha=0.5)
ax.set_xlabel('date')
ax.set_ylabel('next day fractional price change')
ax.set_title('LSTM predictions & actuals, for ethereum')
ax.set_ylim(-0.25, 0.25)
ax.legend()
plt.savefig('figs/prediction.png')
#plt.show()

#LSTM loss function evaluated for testing sample < training loss...probably due to overfitting 
loss_test = np.abs(y_test - y_test_pred.values.flatten())[0:-1].mean()
print 'LSTM loss for testing sample = ', loss_test
loss_test = np.abs(y_test - y_test_pred_LR)[0:-1].mean()
print ' LR  loss for testing sample = ', loss_test

#done!
print "ethereum's predicted long-term daily gain =", y_test_pred[40:].values.mean()
time_stop = time.time()
print 'execution time (minutes) = ', (time_stop - time_start)/60.0

