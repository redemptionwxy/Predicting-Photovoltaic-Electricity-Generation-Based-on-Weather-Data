import numpy
import matplotlib.pyplot as plt
from keras import backend as K
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import numpy as np
import csv


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

dataset = np.loadtxt("YMCA5.csv", delimiter=",", skiprows=1)
dataset = dataset.astype('float32')
# normalize the dataset
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)
bestscore = 0
avscore = 0
avRMSE = 0 
look_back = 5
testamount = 2000
trainX, trainY = dataset[:-testamount, 0:5], dataset[:-testamount, 5]
testX, testY = dataset[-testamount:, 0:5], dataset[-testamount:, 5]

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

for loop in range (1,21):
	model = Sequential()
	model.add(LSTM(8, input_shape=(5, 1)))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=[root_mean_squared_error])
	model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=2)
	model.summary()

	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)
	score = metrics.r2_score(testY,testPredict)
	RMSE = np.sqrt(metrics.mean_squared_error(testY, testPredict))

	print('%d model:'%loop)
	print('R^2 is %f'%score)
	print ('RMSE is %s'%RMSE)

	avscore += score
	avRMSE += RMSE

	if score > bestscore:
		bestresult = testPredict
		bestscore = score
		bestrmse = RMSE
		bestX_train = trainX
		besty_train = trainY
		bestX_test = testX
		besty_test = testY

avscore /= loop
avRMSE /= loop
print ('--------------------------')
print ('Best RMSE is %s'%bestrmse)
print ('Average R^2 is %s'%avscore)
print ('Average RMSE is %s'%avRMSE)

fig, ax = plt.subplots()
plt.style.use('bmh')
test_data_seq = np.arange(1, testamount + 1, 1).tolist()
plt.plot(test_data_seq, besty_test, 'go-', label='true value', color='#826B48')
plt.plot(test_data_seq, bestresult, 'go-', label='predict value', color='#EA5964')
plt.title('Predict PV Electricity Generation Model Score: %f'%bestscore)
ax.set_ylabel('Electricity Generation')
ax.set_xlabel('Hours')
plt.legend()
plt.show()


# path = "./LSTM.csv"
# csvFile = open(path, "w+",newline='')

# try:
#     writer = csv.writer(csvFile)
#     writer.writerow(bestresult)
#     writer.writerow(besty_test)
# finally:
#     csvFile.close()