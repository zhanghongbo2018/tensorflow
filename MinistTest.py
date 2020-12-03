# __author__:HongBo Zhang
# 10/11/2020 上午 11:34
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu', autocast=False, dtype=tf.float32)
        self.flatten = Flatten(autocast=False)
        self.d1 = Dense(128, activation='relu', autocast=False, dtype=tf.float32)
        self.d2 = Dense(10, activation='softmax', autocast=False, dtype=tf.float32)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        #自定义优化器
        self.optimizer = tf.keras.optimizers.SGD()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # 训练步骤
    @tf.function
    def train_step_(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self(images)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    # 测试步骤
    @tf.function
    def test_step_(self, images, labels):
        predictions = self(images)
        t_loss = self.loss_object(labels, predictions)
        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    def test(self):
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()
        for test_images, test_labels in self.test_ds:
            self.test_step_(test_images, test_labels)
        template = 'Test Loss: {}, Test Accuracy: {}'
        print(template.format(
            self.test_loss.result(),
            self.test_accuracy.result() * 100))
        return self.test_accuracy.result() * 100

    # 具体的计算call
    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

    # 训练
    def train_test(self, epoch_data_weight, index):
        for epoch in range(epoch_data_weight):
            # 在下一个epoch开始时，重置评估指标
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            # self.test_loss.reset_states()
            # self.test_accuracy.reset_states()

            for images, labels in self.train_ds:
                self.train_step_(images, labels)
            # 先去掉测试集
            # for test_images, test_labels in self.test_ds:
            #     self.test_step_(test_images, test_labels)

            template = 'index {} Epoch {}, Loss: {}, Accuracy: {}'
            print(template.format(index, epoch + 1,
                                  self.train_loss.result(),
                                  self.train_accuracy.result() * 100))

    def get_weights(self):
        '''

        :return: data_listq
        '''
        weight = []
        weight.append(self.conv1.get_weights())
        weight.append(self.d1.get_weights())
        weight.append(self.d2.get_weights())
        return weight
        # 对应每层的权重设置

    def set_weights(self, weight):
        self.conv1.set_weights(weights=weight[0])
        self.d1.set_weights(weights=weight[1])
        self.d2.set_weights(weights=weight[2])

    # 为每个模型设置数据集
    def set_data(self, train_ds, test_ds=None):
        self.train_ds = train_ds
        self.test_ds = test_ds


# 训练一次，拿到权重
def model_train(model):
    model_weight = []
    for mi, m in enumerate(model):
        m.train_test(epoch_data_weight=1, index=mi + 1)
        model_weight.append(m.get_weights())
    return model_weight


# 交换权重
def set_model_weights(model, model_weight):
    for i, m in enumerate(model):
        m.set_weights(model_weight[len(model) - 1 - i])


# 将权重平均，不做任何处理
def means_weight(weight):
    new_weight = []
    for i in range(3):
        m = np.array(weight[0][i])
        for j in range(1, len(weight)):
            m = np.array(weight[j][i]) + m
        m = m / len(weight)
        new_weight.append(m)
    return new_weight


# 将权重洗牌
def rand_weight(weight):
    new_weight = [[] for i in range(len(weight))]
    layer_weight = [[] for i in range(len(weight[0]))]
    for i in range(len(layer_weight)):
        for j in range(len(weight)):
            layer_weight[i].append(weight[j][i])
    for i in range(len(layer_weight)):
        shuffle(x=layer_weight[i])
    for j in range(len(layer_weight)):
        for i in range(len(layer_weight[0])):
            new_weight[i].append(layer_weight[j][i])
    return new_weight


# 设置权重
def set_rand_weight(model, weight):
    for mi, m in enumerate(model):
        m.set_weights(weight[mi])


# 权重的求和，权值是和数据量为正相关
def weight_means_sum(weight, data_weight):
    '''

    :param weight: 传回来的模型权重
    :param data_weight: 每个模型的数据权重
    :return: 返回新的权重
    '''
    new_weight = []
    for i in range(3):
        m = np.array(weight[0][i]) * data_weight[0]
        for j in range(1, len(weight)):
            m = m + np.array(weight[j][i]) * data_weight[j]
        new_weight.append(m)
    return new_weight


# 设置权重
def set_means(weight, model):
    for m in model:
        m.set_weights(weight)


# 数据处理以及初始化模型，这个是分类处理数据
def init_model_class(x_train, y_train):
    # data_list用来切片，获取对应每个类的数据
    data_list = [[] for i in range(10)]
    for yi, y in enumerate(y_train):
        data_list[y].append(x_train[yi])
    # 获取测试集数据总量
    total_data_num = x_train.shape[0]
    # 用来求数据权重
    data_weight = []
    for i in range(len(data_list)):
        data_list[i] = np.array(data_list[i])
        data_weight.append(data_list[i].shape[0] / total_data_num)
    data_weight = np.array(data_weight, dtype=np.float)
    data_list = np.array(data_list)
    # 用来保存模型
    model_total = []
    for i in range(len(data_list)):
        model = MyModel()
        train_ds = tf.data.Dataset.from_tensor_slices(
            (data_list[i], np.array([i for j in range(data_list[i].shape[0])]))).shuffle(10000).batch(32)
        model.set_data(train_ds=train_ds, test_ds=None)
        model_total.append(model)
    return model_total, data_weight


# 均匀切割数据，并且初始化模型
def init_model_means_data(x_train, y_train):
    model_total = []
    x = [0, 10001, 20001, 30001, 40001, 50001]
    for i in x:
        train_ds = tf.data.Dataset.from_tensor_slices(
            (x_train[i:i + 10000], y_train[i:i + 10000])).shuffle(10000).batch(32)
        model = MyModel()
        model.set_data(train_ds, test_ds=None)
        model_total.append(model)
    return model_total


# 数据处理 归一化
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)
# 开始训练
# 模拟中心节点的模型
model = MyModel()
model.set_data(train_ds=train_ds)
# 专门用来构建模型
model.train_test(epoch_data_weight=1, index=0)
# 获取测试集
x_y_test_ds = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).shuffle(10000).batch(32)
model.set_data(train_ds=train_ds, test_ds=x_y_test_ds)
# 最后用来保存权重的数组，平均之后的权重
weight_single = []
# 保存模型的准确率
result = []
model_total, data_weight = init_model_class(x_train, y_train=y_train)
# model_total = init_model_means_data(x_train, y_train)
weight = []
for i in range(20):
    print("-----------------{}-----------------------".format(i))
    weight = model_train(model_total)
    # 将权重线性求和再平均
    # weight_single = weight_means_sum(weight=weight, data_weight=data_weight)
    # weight_single = means_weight(weight)
    # set_means(weight=weight_single, model=model_total)
    weight = rand_weight(weight)
    set_rand_weight(model_total, weight)
    weight = weight_means_sum(weight, data_weight)
    model.set_weights(weight)
    # model.test()
    # model.set_weights(weight[0])
    r = model.test()
    result.append(r)
    print("-----------------END-----------------------")
# weight = weight_means_sum(weight, data_weight)
# model.set_weights(weight)
# model.test()
#####################################################################
list = []
for yi, y in enumerate(y_test):
    if y == 1:
        list.append(x_test[yi])
list = np.array(list)
x_y_test_ds_1 = tf.data.Dataset.from_tensor_slices(
    (list, np.array([1 for i in range(list.shape[0])]))).shuffle(10000).batch(32)
model.set_data(train_ds=None, test_ds=x_y_test_ds_1)
model.test()

print(result)
plt.plot([i for i in range(len(result))], result, color="blue", linewidth=3)
plt.grid(True)
plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
plt.show()
