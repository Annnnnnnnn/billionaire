import obtainNews
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense
from sklearn import preprocessing
import matplotlib.pyplot as plt

news_vector = obtainNews.obtain_news("2019/4/15")
onehot_vector = obtainNews.one_hot_encoding(news_vector)
predictive_value = np.array([30.14525, 30.37803, 30.69989, 31.06778, 31.20263, 31.25064, 30.83452, 30.44287, 30.42028, 30.31045, 30.1874 , 30.00289, 30.05544, 29.62501, 29.56649, 29.27254, 28.99363, 29.16610, 29.36760, 29.13651, 29.34064, 29.56951, 29.8315])
real_value =  np.array([30.700001, 31.16, 31.73, 31.6, 31.73, 31.030001, 31.01, 30.67, 30.68, 30.41, 30.35, 30.129999, 29.620001, 29.940001, 29.200001, 29.15, 29.15, 29.370001, 29.059999, 29.540001, 29.690001, 30.01, 30.040001])

model = Sequential()
model.add(Dense(input_dim=2, units=1))
model.add(Dense(input_dim=1, units=1))
model.compile(loss='mse', optimizer='adam')
model.summary()

X_train = predictive_value
X_train = np.stack((predictive_value, onehot_vector), axis=1)
Y_train = real_value

model.fit(X_train, Y_train, batch_size=1, epochs=15)
result = model.predict(X_train)

plt.plot(Y_train, color='green', label='real')
plt.plot(result, color='red', label='predictive')
plt.show()