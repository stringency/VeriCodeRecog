import tensorflow as tf
from tensorflow.keras.layers import Dense,GlobalAvgPool2D,Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import EarlyStopping,CSVLogger,ModelCheckpoint,ReduceLROnPlateau
import string
import numpy as np
import os
from plot_model import plot_model
# 字符包含所有数字和所有小写英文字母，一共 62 个
characters = string.digits + string.ascii_letters
# 类别数 62
num_classes = len(characters)
# 批次大小
batch_size = 64
# 周期数
epochs=100
# 训练集数据，大约 50000 张图片
# 事先用 captcha 模块生成，长度都是 4
train_dir = "./captcha/train/"
# 测试集数据，大约 10000 张图片
# 事先用 captcha 模块生成，长度都是 4
test_dir = "./captcha/test/"
# 图片宽度
width=160
# 图片高度
height=60
# 获取所有验证码图片路径和标签
def get_filenames_and_classes(dataset_dir):
    # 存放图片路径
    photo_filenames = []
    # 存放图片标签
    y = []
    for filename in os.listdir(dataset_dir):
        # 获取文件完整路径
        path = os.path.join(dataset_dir, filename)
        # 保存图片路径
        photo_filenames.append(path)
        # 取文件名前 4 位，也就是验证码的标签
        captcha_text = filename[0:4]
        # 定义一个空 label
        label = np.zeros((4, num_classes), dtype=np.uint8)
        # 标签转独热编码
        for i, ch in enumerate(captcha_text):
            # 设置标签，独热编码 one-hot 格式
            # characters.find(ch)得到 ch 在 characters 中的位置，可以理解为 ch 的编号
            label[i, characters.find(ch)] = 1
        # 保存独热编码的标签
        y.append(label)
    # 返回图片路径和标签
    return np.array(photo_filenames),np.array(y)
# 获取训练集图片路径和标签
x_train,y_train = get_filenames_and_classes(train_dir)
# 获取测试集图片路径和标签
x_test,y_test = get_filenames_and_classes(test_dir)
# 图像处理函数
# 获得每一条数据的图片路径和标签
def image_function(filenames, label):
    # 根据图片路径读取图片内容
    image = tf.io.read_file(filenames)
    # 将图像解码为 jpeg 格式的 3 维数据
    image = tf.image.decode_jpeg(image, channels=3)
    # 归一化
    image = tf.cast(image, tf.float32) / 255.0
    # 返回图片数据和标签
    return image, label
# 标签处理函数
# 获得每一个批次的图片数据和标签
def label_function(image, label):
    # transpose 改变数据的维度，比如原来的数据 shape 是(64,4,62)
    # 这里的 64 是批次大小，验证码长度为 4 有 4 个标签，62 是 62 个不同的字符
    # tf.transpose(label,[1,0,2])计算后得到的 shape 为(4,64,62)
    # 原来的第 1 个维度变成了第 0 维度，原来的第 0 维度变成了 1 维度，第 2 维不变
    # (64,4,62)->(4,64,62)
    label = tf.transpose(label, [1, 0, 2])
    # 返回图片内容和标签，注意这里标签的返回，我们的模型会定义 4 个任务，所以这里返回4个标签
    # 每个标签的 shape 为(64,62)，64 是批次大小，62 是独热编码格式的标签
    return image, (label[0], label[1], label[2], label[3])
# 创建 dataset 对象，传入训练集图片路径和标签
dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# 打乱数据，buffer_size 定义数据缓冲器大小，随意设置一个较大的值
# reshuffle_each_iteration=True，每次迭代都会随机打乱
dataset_train = dataset_train.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
# map-可以自定义一个函数来处理每一条数据
dataset_train = dataset_train.map(image_function)
# 数据重复生成 1 个周期
dataset_train = dataset_train.repeat(1)
# 定义批次大小
dataset_train = dataset_train.batch(batch_size)
# 注意这个 map 和前面的 map 有所不同，第一个 map 在 batch 之前，所以是处理每一条数据
# 这个 map 在 batch 之后，所以是处理每一个 batch 的数据
dataset_train = dataset_train.map(label_function)
# 创建 dataset 对象，传入测试集图片路径和标签
dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# 打乱数据，buffer_size 定义数据缓冲器大小，随意设置一个较大的值
# reshuffle_each_iteration=True，每次迭代都会随机打乱
dataset_test = dataset_test.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
# map-可以自定义一个函数来处理每一条数据
dataset_test = dataset_test.map(image_function)
# 数据重复生成 1 个周期
dataset_test = dataset_test.repeat(1)
# 定义批次大小
dataset_test = dataset_test.batch(batch_size)
# 注意这个 map 和前面的 map 有所不同，第一个 map 在 batch 之前，所以是处理每一条数据
# 这个 map 在 batch 之后，所以是处理每一个 batch 的数据
dataset_test = dataset_test.map(label_function)
# 生成一个批次的数据和标签
# 可以用于查看数据和标签的情况
x,y = next(iter(dataset_test))
print(x.shape)
print(np.array(y).shape)


# 也可以使用循环迭代的方式循环一个周期的数据，每次循环获得一个批次
# for x,y in dataset_test:
# pass
# 载入预训练的 resnet50 模型
resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(height,width,3))
# resnet50 = ResNet50('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
# 设置输入
inputs = Input((height,width,3))
# 使用 resnet50 进行特征提取
x = resnet50(inputs)
# 平均池化
x = GlobalAvgPool2D()(x)
# 把验证码识别的 4 个字符看成是 4 个不同的任务
# 每个任务负责识别 1 个字符
# 任务 1 识别第 1 个字符，任务 2 识别第 2 个字符，任务 3 识别第 3 个字符，任务 4 识别第4 个字符
x0 = Dense(num_classes, activation='softmax', name='out0')(x)
x1 = Dense(num_classes, activation='softmax', name='out1')(x)
x2 = Dense(num_classes, activation='softmax', name='out2')(x)
x3 = Dense(num_classes, activation='softmax', name='out3')(x)
# 定义模型
model = Model(inputs, [x0,x1,x2,x3])
# 画图
''' 
plot_model(model,style=0)

修改备注：网上说由于2.8开始就没有了一套的形式，模块发生转变了，需要在其它地方拿出来
'''
from tensorflow.keras.utils import plot_model
plot_model(model)
# 4 个任务我们可以定义 4 个 loss
# loss_weights 可以用来设置不同任务的权重，验证码识别的 4 个任务权重都一样
model.compile(loss={'out0':'categorical_crossentropy',
'out1':'categorical_crossentropy',
'out2':'categorical_crossentropy',
'out3':'categorical_crossentropy'},
loss_weights={'out0':1,
'out1':1,
'out2':1,
'out3':1},
optimizer=SGD(lr=1e-2,momentum=0.9),
metrics=['acc'])
# 监控指标统一使用 val_loss
# 可以使用 EarlyStopping 来让模型停止，连续 6 个周期 val_loss 没有下降就结束训练
# CSVLogger 保存训练数据
# ModelCheckpoint 保存所有训练周期中 val_loss 最低的模型
# ReduceLROnPlateau 学习率调整策略，连续 3 个周期 val_loss 没有下降当前学习率乘以0.1
callbacks = [EarlyStopping(monitor='val_loss', patience=6, verbose=1),
    CSVLogger('Captcha_tfdata.csv'),
    ModelCheckpoint('Best_Captcha_tfdata.h5', monitor='val_loss', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)]
# 训练模型
# 把之前定义的 dataset_train 和 dataset_test 传入进行训练
model.fit(x=dataset_train,epochs=epochs,validation_data=dataset_test,callbacks=callbacks)