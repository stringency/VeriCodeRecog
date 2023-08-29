from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input,Dense,GlobalAvgPool2D
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.callbacks import EarlyStopping,CSVLogger,ModelCheckpoint,ReduceLROnPlateau
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
import string
from plot_model import plot_model
# 字符包含所有数字和所有大小写英文字母，一共 62 个
characters = string.digits + string.ascii_letters
# 类别数
num_classes = len(characters)
# 批次大小
batch_size = 64
# 训练集批次数
# 训练集大小相当于是 64*1000=64000
train_steps = 1000
# 测试集批次数
# 测试集大小相当于是 64*100=6400
test_steps = 100
# 周期数
epochs=100
# 图片宽度
width=160
# 图片高度
height=60
# 用于自定义数据生成器
from tensorflow.keras.utils import Sequence
# 这里的 Sequence 定义其实不算典型，因为一般的数据集数量是有限的，
# 把所有数据训练一次属于训练一个周期，一个周期可以分为 n 个批次，
# Sequence 一般是定义一个训练周期内每个批次的数据如何产生。
# 我们这里的验证码数据集使用 captcha 模块生产出来的，一边生产一边训练，可以认为数据集是无限的。
class CaptchaSequence(Sequence):
    # __getitem__和__len__是必须定义的两个方法
    def __init__(self, characters, batch_size, steps, n_len=4, width=160, height=60):
        # 字符集
        self.characters = characters
        # 批次大小
        self.batch_size = batch_size
        # 生成器生成多少个批次的数据
        self.steps = steps
        # 验证码长度
        self.n_len = n_len
        # 验证码图片宽度
        self.width = width
        # 验证码图片高度
        self.height = height
        # 字符集长度
        self.num_classes = len(characters)
        # 用于产生验证码图片
        self.image = ImageCaptcha(width=self.width, height=self.height)
        # 用于保存最近一个批次验证码字符
        self.captcha_list = []
    # 获得 index 位置的批次数据
    def __getitem__(self, index):
        # 初始化数据用于保存验证码图片
        x = np.zeros((self.batch_size, self.height, self.width, 3), dtype=np.float32)
        # 初始化数据用于保存标签
        # n_len 是多任务学习的任务数量，这里是 4 个任务，batch 批次大小，num_classes 分类数量
        y = np.zeros((self.n_len, self.batch_size, self.num_classes), dtype=np.uint8)
        # 数据清 0
        self.captcha_list = []
        # 生产一个批次数据
        for i in range(self.batch_size):
            # 随机产生验证码
            captcha_text = ''.join([random.choice(self.characters) for j in range(self.n_len)])
            self.captcha_list.append(captcha_text)
            # 生产验证码图片数据并进行归一化处理
            x[i] = np.array(self.image.generate_image(captcha_text)) / 255.0
            # j(0-3),i(0-61),ch(单个字符)
            # self.characters.find(ch)得到 c 在 characters 中的位置，可以理解为 c 的编号
            for j, ch in enumerate(captcha_text):
                # 设置标签，独热编码 one-hot 格式
                y[j, i, self.characters.find(ch)] = 1
        # 返回一个批次的数据和标签
        return x, [y[0],y[1],y[2],y[3]]
    # 返回批次数量
    def __len__(self):
        return self.steps
# 测试生成器
# 一共一个批次，批次大小也是 1
data = CaptchaSequence(characters, batch_size=1, steps=1)
for i in range(2):
    # 产生一个批次的数据
    x, y = data[0]
    # 显示图片
    plt.imshow(x[0])
    # 验证码字符和对应编号
    plt.title(data.captcha_list[0])
    plt.show()


# 载入预训练的 resnet50 模型
resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(height,width,3))
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
CSVLogger('Captcha.csv'),
ModelCheckpoint('Best_Captcha.h5', monitor='val_loss', save_best_only=True),
ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)]
# 训练模型
model.fit(x=CaptchaSequence(characters, batch_size=batch_size, steps=train_steps),
epochs=epochs,
validation_data=CaptchaSequence(characters, batch_size=batch_size, steps=test_steps),
callbacks=callbacks)









