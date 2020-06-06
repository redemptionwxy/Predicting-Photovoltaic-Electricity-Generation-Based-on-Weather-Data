import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
import csv

bestscore = 0
avscore = 0
avRMSE = 0
testamount = 2000
dataset = np.loadtxt("YMCA5.csv", delimiter=",", skiprows=1)

for loop in range (1,51):
	np.random.shuffle(dataset)

	X_train, y_train = dataset[:-testamount, 0:5], dataset[:-testamount, 5]
	X_test, y_test = dataset[-testamount:, 0:5], dataset[-testamount:, 5]

	model = MLPRegressor(solver='adam', hidden_layer_sizes=(16, 32, 64 )
		, activation='logistic',max_iter=200, learning_rate='constant')
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

avscore /= loop
avRMSE /= loop
print ('--------------------------')
print ('Best R^2 id %f'%bestscore)
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