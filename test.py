import obtainNews
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense
from sklearn import preprocessing

news_vector = obtainNews.obtain_news("2019/4/15")
onehot_vector = obtainNews.one_hot_encoding(news_vector)
predictive_value = np.array([1,2,3,4,5,6,7,8,9,0,9,8,7,6,5,4,3,2,1,2,3,4,5])
real_value =  np.array([0,9,8,7,6,7,8,9,4,5,3,5,6,7,8,3,4,5,6,98,2,4,3])

model = Sequential()
model.add(Dense(input_dim=2, units=1))
model.add(Dense(input_dim=1, units=1))
model.compile(loss='mse', optimizer='adam')
model.summary()

X_train = np.stack((predictive_value, onehot_vector), axis=1)
print(X_train)
Y_train = real_value
print(X_train)
model.fit(X_train, Y_train, batch_size=1, epochs=10)
result = model.predict(X_train)
print(result)