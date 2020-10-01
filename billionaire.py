import matplotlib.pyplot as plt
import os
import  math
import pandas as pd
import numpy as np
import keras
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import getStockData as web

# 定义滑动窗口大小
seq_len = 22
# 定义神经网络层数
neurous = [128, 128, 32, 1]
# 定义迭代此时
epochs = 200
# 定义输入维度
shape = [4, seq_len, 1]
# 数据文件存储位置
BASE_PATH = 'historical stocks data\\'
STOCK_NAME = 'NFLX'
path = BASE_PATH + STOCK_NAME


def get_stock_data(stock_name, normalize=True):
    """
    获取指定股票的开盘、当日最高、当日最低和调整后的收盘价
    可以选择是否进行正则化
    """
    df = web.get_stock_data_from_web(stock_name,2019,2,14,2020,2,14)
    df.drop(['Volume','Close'],1,inplace=True)

    if normalize:
        scaler = preprocessing.MinMaxScaler()
    
    df['Open'] = scaler.fit_transform(df.Open.values.reshape(-1,1))
    df['High'] = scaler.fit_transform(df.High.values.reshape(-1,1))
    df['Low'] = scaler.fit_transform(df.Low.values.reshape(-1,1))
    df['Adj Close'] = scaler.fit_transform(df['Adj Close'].values.reshape(-1,1))
    return df

def plot_stock(stock_name):
    """
    绘制指定股票 Adj Close 价格曲线
    """
    df = get_stock_data(stock_name, normalize=True)
    plt.plot(df['Adj Close'], color='red', label='Adj Close')
    plt.legend(loc='best')
    plt.show()

def load_data(stock, seq_len):
    """
    将数据分割为训练集和测试集
    将每一个集合分割为用于预测的X和预测结果Y
    使用滑动窗口的方式分割数据集，窗口的最后一个单元值最为Y
    """
    amount_of_features = len(stock.columns)
    data = stock.values
    sequence_length = seq_len + 1
    result = []
    # 设置窗口，一共有 “最新日期-窗口日期长度” 个窗口。将每一个窗口放入result数组中
    for index in range(len(data) - sequence_length):
       result.append(data[index: index + sequence_length])

    result = np.array(result)
    row = round(0.9 * result.shape[0])

    train = result[:int(row),:]
    X_train = train[:,:-1]
    Y_train = train[:, -1][:,-1]

    X_test = result[int(row):,:-1]
    Y_test = result[int(row):, -1][:,-1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))

    return [X_train, Y_train, X_test, Y_test]

def build_LSTM_model(layers, neurous, d):
    """
    建立两层LSTM神经网络模型
    每一层之间为了防止过拟合舍弃一些节点
    添加两个全连接层
    """
    model = Sequential()
    model.add(LSTM(neurous[0], input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
    model.add(LSTM(neurous[1], input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))
    model.add(Dense(neurous[2], kernel_initializer='uniform', activation='relu'))
    model.add(Dense(neurous[3], kernel_initializer='uniform', activation='linear'))
    adam = keras.optimizers.Adam(decay=0.2)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def visualize_loss(history):
    """
    将训练和测试过程中的损失函数可视化
    """
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

def model_score(model, X_train, Y_train, X_test, Y_test):
    """
    使用两个标准: MSE RMSE 均方误差和均方误差根来对预测结果进行打分
    """
    train_score = model.evaluate(X_train, Y_train, verbose=0)
    print('Train Score: %.5f MSE (%.2f RMSE)' % (train_score[0], math.sqrt(train_score[0])))

    test_score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test Score: %.5f MSE (%.2f RMSE)' % (test_score[0], math.sqrt(test_score[0])))
    return train_score, test_score

def percentage_difference(model, X_test, Y_test):
    """
    比较预测值和真实值的差距
    暂不反悔比较差距，返回预测值即model predict value
    """
    percentage_diff = []
    predictive_value = model.predict(X_test)
    # for i in range(len(Y_test)):
    #     pi = predictive_value[i][0]
    #     percentage_diff.append((abs(pi-Y_test[i])/pi)*100)
    return predictive_value

def visualize_normalized_result(stock_name, predictive_value, Y_test):
    plt.plot(predictive_value, color='red', label='Prediction')
    plt.plot(Y_test, color='green', label='Real')
    plt.legend(loc='best')
    plt.title("Prediction result for stock {}".format(stock_name))
    plt.xlabel('Days')
    plt.ylabel('Adj Close')
    plt.show()

def denormalize(stock_name, normalized_value):
    """
    逆归一化
    """
    df = pd.read_csv(BASE_PATH+stock_name+'.csv', usecols=['Adj Close']).values.reshape(-1,1)
    normalized_value = normalized_value.reshape(-1,1)

    scaler = preprocessing.MinMaxScaler()
    _ = scaler.fit_transform(df)
    denormalized = scaler.inverse_transform(normalized_value)
    return denormalized

def visualize_result(stock_name, normalized_pridictive_value, normalized_Y_test):
    """
    将逆归一化后的结果可视化
    """
    denormalized_predictive_value = denormalize(stock_name, normalized_pridictive_value)
    denormalized_Y_test = denormalize(stock_name, normalized_Y_test)
    print("predictive value: ", denormalized_predictive_value)
    print("real value: ", denormalized_Y_test)
    plt.plot(denormalized_predictive_value, color='red', label='Prediction')
    plt.plot(denormalized_Y_test, color='green', label='Real')
    plt.legend(loc='best')
    plt.title("Prediction result for stock {}".format(stock_name))
    plt.xlabel('Days')
    plt.ylabel('Adj Close')
    plt.show()

def quick_measure(stock_name, seq_len, d, shape, neurous, epochs):
    """
    检测函数 
    对不断优化的模型训练准确度进行打分
    """
    df = get_stock_data(stock_name)
    X_train, Y_train, X_test, Y_test = load_data(df, seq_len)
    model = build_LSTM_model(shape, neurous, d)

    model.fit(X_train, Y_train, batch_size=126, epochs=epochs, verbose=1, validation_split=0.1)
    train_score, test_score = model_score(model, X_train, Y_train, X_test, Y_test)
    return train_score, test_score

def test_optimizer():
    """
    测试优化dropout参数的数值
    经过计算发现0.22所对应的结果是最好的
    但是这个不是最终的结果 还需要根据其他参数的调整伴随调整
    """
    d_list = [0.22, 0.225, 0.23, 0.235, 0.24]
    dorpout_result = {}

    for d in d_list:
        train_score, test_score = quick_measure(STOCK_NAME, seq_len, d, shape, neurous, epochs)
        dorpout_result[d] = test_score

    min_val = min(dorpout_result.values())
    min_val_key = [k for k,v in dorpout_result.items() if v== min_val]
    print(min_val_key)
    print(dorpout_result)
    return ()

def neurous_optimizer(stock_name):
    seq_len = 22
    # feature window output
    shape = [4, seq_len, 1]
    epochs = 100
    dropout = 0.22
    neurous_list1 = [128,256,512,1024,2048]
    neurous_list2 = [16, 32, 64]
    neurous_result = {}

    for neurous_lstm in neurous_list1:
        # 前两层LSTM神经元个数
        neurous = [neurous_lstm, neurous_lstm]
        for activation in neurous_list2:
            # 第三层全连接层的神经元个数
            neurous.append(activation)
            # 最后一层输出的神经元个数
            neurous.append(1)
            train_score, test_score = quick_measure(stock_name, seq_len, dropout, shape, neurous, epochs)
            neurous_result[str(neurous)] = test_score
            neurous = neurous[:2]
    return neurous_result

def visualize_predictive_result_with_real_result(stock_name):
    """
    将预测结果与真实结果对比可视化
    """
    df = get_stock_data(stock_name)
    X_train, Y_train, X_test, Y_test = load_data(df, seq_len)
    model = build_LSTM_model(shape, neurous, 0.1)
    history = model.fit(X_train, Y_train, batch_size=50, epochs=epochs, validation_split=0.1, verbose=1)
    model_score(model, X_train, Y_train, X_test, Y_test)
    predictive_value = percentage_difference(model, X_test, Y_test)
    visualize_result(stock_name, predictive_value, Y_test)

def main(stock_name):
    df = get_stock_data(stock_name)
    X_train, Y_train, X_test, Y_test = load_data(df, seq_len)
    model = build_LSTM_model(shape, neurous, 0.1)
    history = model.fit(X_train, Y_train, batch_size=50, epochs=epochs, validation_split=0.1, verbose=1)
    # visualize_loss(history)
    model_score(model, X_train, Y_train, X_test, Y_test)
    predictive_value = percentage_difference(model, X_test, Y_test)
    return predictive_value

visualize_predictive_result_with_real_result("TGT")

