import numpy as np
from keras.models import Sequential 
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn import preprocessing
import obtainNews 


scaler = preprocessing.MinMaxScaler()
predictive_value = np.array([334.7023 ,335.67844,   337.7035 ,338.79623,339.60962,339.9784 ,339.80237,   337.50577,   338.26804,   343.4581 ,   346.1367 ,   348.43906,   349.52313,   349.96622,   350.45065,352.8684 ,357.61255,362.71948,365.5412 ,367.1552 ,368.6186 ,371.04428,373.99933])
real_value =  np.array([338.92001345,  338.69000246,339.07000,338.61999,339.67001,338.10998,325.99999997,349.60000617,353.16000375,342.88000492,348.51998908,343.1600037 ,347.7399903 ,345.08999639,358.0000 ,369.01000,369.67001,366.95001,366.76998,371.07000,373.69000,380.01000,381.39999])
one_hot_feature=np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
predictive_profit = []
real_profit = []


buy = 100*predictive_value[0]
for y in range(2, 24):
    profit = 100*predictive_value[y-1] - buy
    predictive_profit.append(profit)

buy = 100*real_value[0]
for y in range(2, 24):
    profit = 100*real_value[y-1] - buy
    real_profit.append(profit)


# predictive_profit = scaler.fit_transform(predictive_value.reshape(-1,1))
# real_profit = scaler.fit_transform(real_value.reshape(-1,1))

predictive_profit = np.stack((predictive_profit, one_hot_feature), axis=1)


X_train = predictive_profit[:20]
Y_train = real_profit[:20]
X_test = predictive_profit[20:]
Y_test = real_profit[20:]

model = Sequential()
model.add(Dense(input_dim=2, units=1))
model.add(Dense(input_dim=1, units=1))
model.compile(loss='mse', optimizer='adam')
model.summary()

print("Training...........")
# for step in range(500):
#     cost = model.train_on_batch(X_train, Y_train)
#     if(step%10 == 0):
#         print("cost: ", cost)

print("Testing...........")
cost = model.evaluate(X_test, Y_test, batch_size=4)
W, b = model.layers[0].get_weights()
print("Weight:",W,'\nbiases=', b)

predictive = model.predict(X_train)

for i in range(len(Y_train)):
    print(Y_train[i] - predictive[i])

plt.plot(Y_train, color='green', label='real')
plt.plot(predictive, color='red', label='predictive')
plt.show()

