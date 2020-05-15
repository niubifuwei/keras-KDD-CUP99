import numpy as np
import time
start = time.time()
import keras
from keras.models import Sequential  #序贯模型
from keras.layers import Dense    #全连接层
from keras.layers import Dropout  #随机失活层
from keras.layers import Flatten  #展平层，从卷积层到全连接层必须展平
from keras.layers import Conv1D   #卷积层
from keras.layers import MaxPooling1D  #最大值池化
import pandas as pd
from keras import backend as k
#from sklearn.cross_validation import train_test_split #随机划分为训练子集和测试子集
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD
import matplotlib.pyplot as plt
#f = pd.read_csv('/home/tianchi/myspace/KDDTrain+number.csv', header=None)
x_train= np.loadtxt("KDDTrain+number.csv",delimiter=",",usecols=np.arange(0,41))
#print(x_train[0])
y_train=np.loadtxt("KDDTrain+number.csv",delimiter=",",usecols=np.arange(41,42))
#print(y_train)
x_test= np.loadtxt("KDDTest+number.csv",delimiter=",",usecols=np.arange(0,41))
#print(x_train[0])
y_test=np.loadtxt("KDDTest+number.csv",delimiter=",",usecols=np.arange(41,42))
batch_size = 128  #一批训练样本128张图片
num_classes = 2  #有2个类别
epochs = 12   #一共迭代12轮

if k.image_data_format() == 'channels_first':
   x_train = x_train.reshape(x_train.shape[0], 1, 41)
   x_test = x_test.reshape(x_test.shape[0], 1, 41)
   #x_dev = x_dev.reshape(x_dev.shape[0], 1, 41)
   input_shape = (1, 41)
else:
   x_train = x_train.reshape(x_train.shape[0], 41, 1)
   x_test = x_test.reshape(x_test.shape[0], 41, 1)
   #x_dev = x_dev.reshape(x_dev.shape[0], 41, 1)
   input_shape = (41, 1)

model = Sequential()  #sequential序贯模型:多个网络层的线性堆叠
#输出的维度（卷积滤波器的数量）filters=32；1D卷积窗口的长度kernel_size=3；激活函数activation   模型第一层需指定input_shape：
model.add(Conv1D(32, 3, activation='relu',input_shape=input_shape))  #data_format默认channels_last
model.add(MaxPooling1D(pool_size=(2))) #池化层：最大池化  池化窗口大小pool_size=2

model.add(Flatten())  #展平一个张量，返回一个调整为1D的张量
#model.add(Dropout(0.25))  #需要丢弃的输入比例=0.25    dropout正则化-减少过拟合
model.add(Dense(128, activation='relu',name='fully_connected')) #全连接层
model.add(Dense(1, activation='sigmoid',name='sigmoid'))

#编译，损失函数:多类对数损失，用于多分类问题， 优化函数：adadelta， 模型性能评估是准确率
model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
#运行 ， verbose=1输出进度条记录      epochs训练的轮数     batch_size:指定进行梯度下降时每个batch包含的样本数
#history=model.fit(x_train, y_train, batch_size= batch_size, epochs=epochs, verbose=1)
history = model.fit(x_train, y_train, batch_size= batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs,loss, 'bo', label = 'Training loss')
plt.plot(epochs,val_loss, 'b', label = 'Validation loss')
plt.title('Training and validatio loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.figure()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc,'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and validatio accuracy')
plt.legend()
plt.show()