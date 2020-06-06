import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn import metrics
import csv
loop = 0
bestscore = 0
avscore = 0
avRMSE = 0
testamount = 240

dataset = np.loadtxt("YMCA4800.csv", delimiter=",", skiprows=1)
train = dataset[:,0:5]
test = dataset[:,5]

kf = KFold(n_splits=20)
kf.get_n_splits(dataset)
for train_index, test_index in kf.split(train,test):
	X_train, X_test = train[train_index], train[test_index]
	y_train, y_test = test[train_index], test[test_index]

	loop = loop + 1 
	
	model = MLPRegressor(solver='adam', hidden_layer_sizes=( 16, 32, 64 )
		, activation='logistic',max_iter=1000, learning_rate='constant')
	model.fit(X_train, y_train)
	result = model.predict(X_test)
	score = model.score(X_test, y_test)
	RMSE = np.sqrt(metrics.mean_squared_error(y_test, result))


	
	print('%d model:'%loop)
	print('R^2 is %f'%score) 
	print ('RMSE is %s'%RMSE)

	avscore += score
	avRMSE += RMSE

	if score > bestscore:
		bestresult = result
		bestscore = score
		bestrmse = RMSE
		bestX_train = X_train
		besty_train = y_train
		bestX_test = X_test
		besty_test = y_test


	# fig, ax = plt.subplots()
	# plt.style.use('bmh')
	# test_data_seq = np.arange(1, testamount + 1, 1).tolist()
	# plt.plot(test_data_seq, y_test, 'go-', label='true value', color='#826B48')
	# plt.plot(test_data_seq, result, 'go-', label='predict value', color='#EA5964')
	# plt.title('Predict PV Electricity Generation Model Score: %f '%score)
	# ax.set_ylabel('Electricity Generation')
	# ax.set_xlabel('Hours')
	# plt.legend()
	# plt.show()


avscore /= loop
avRMSE /= loop
print ('--------------------------')
print ('Best R^2 id %f'%bestscore)
print ('Best RMSE is %s'%bestrmse)
print ('Average R^2 is %s'%avscore)
print ('Average RMSE is %s'%avRMSE)



# cengindex = 0
# for wi in model.coefs_:
#     cengindex += 1  # The layer of nerual network
#     print('\n%d layer of nerual network:' % cengindex)
#     print('weight shape:', wi.shape)
#     print('weight coefficient matrix：\n', wi)


# path = "./file.csv"
# csvFile = open(path, "w+",newline='')

# try:
#     writer = csv.writer(csvFile)
#     writer.writerow(bestresult)
#     writer.writerow(besty_test)
# finally:
#     csvFile.close()