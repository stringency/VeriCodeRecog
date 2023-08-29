from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input,Dense,Reshape,Bidirectional,GRU,Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import string
# 字符包含所有数字和所有大小写英文字母，一共 62 个
characters = string.digits + string.ascii_letters
# 类别数+空白字符
num_classes = len(characters)+1
# 图片宽度
width=160
# 图片高度
height=60
# RNN 的 cell 数量
RNN_cell = 128
# 最长验证码
max_len = 6
# Keras 调用 Tensorflow 中的 ctc_batch_cost
# x 是模型输出，shape-(?,10,63)
# labels 是验证码的标签，shape-(?,max_len)
# input_len 是 x 的长度，shape-(?,1)，x 的长度为 10
# label_len 是 labels 的的长度，shape-(?,1)，labels 的长度为 max_len
def ctc_lambda_func(args):
    x, labels, input_len, label_len = args
    # Tensorflow 中封装的 ctc 计算
    return K.ctc_batch_cost(labels, x, input_len, label_len)
# 载入预训练的 resnet50 模型
resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(height,width,3))
# 设置输入
image_input = Input((height,width,3), name='image_input')
# 使用 resnet50 进行特征提取
x = resnet50(image_input)
# 搭建 RNN 网络
x = Reshape((10,2048))(x)
x = Bidirectional(GRU(RNN_cell, return_sequences=True))(x)
x = Bidirectional(GRU(RNN_cell, return_sequences=True))(x)
x = Dense(num_classes, activation='softmax')(x)
# 定义模型
model = Model(image_input, x)
# 定义标签输入
labels = Input(shape=(max_len), name='max_len')
# 输入长度
input_len = Input(shape=(1), name='input_len')
# 标签长度
label_len = Input(shape=(1), name='label_len')
# Lambda 的作用是可以将自定义的函数封装到网络中，用于自定义的一些数据计算处理
ctc_out = Lambda(ctc_lambda_func, name='ctc')([x, labels, input_len, label_len])
# 定义模型
ctc_model = Model(inputs=[image_input, labels, input_len, label_len], outputs=ctc_out)
# 注意这里是 load_weights，载入权值，这里不能直接用 load_model 载入模型
# 因为 keras 中没有封装 ctc 的 loss，ctc 的 loss 是在 tensorflow 中定义的，属于 keras 外部自定义 loss
# 模型 save 的时候如果包含了自定义 loss，那么在 load_model 的时候也需要声明自定义loss。
# 在这个应用中还是重新搭建一遍模型并使用 load_weights 载入模型权值比较简单
model.load_weights('Best_Captcha_ctc.h5')
# 用于预测的字符集多一个空白符
pre_characters = characters + '-'
# 使用贪心算法预测结果
def greedy(captcha_text):
    # 自定义产生一个验证码
    captcha_text = captcha_text
    # 产生验证码并归一化
    image = ImageCaptcha(width=160, height=60)
    x = np.array(image.generate_image(captcha_text)) / 255.0
    # 变成 4 维数据
    X_test = np.expand_dims(x, axis=0)
    # 用模型进行预测
    y_pred = model.predict(X_test)
    # 查看 y_pred 的 shape
    print("y_pred shape:",y_pred.shape)
    # 获得每个序列最大概率的输出所在位置，其实也就是字符编号
    argmax = np.argmax(y_pred[0], axis=-1)
    print('id','\t','characters')
    for x in argmax:
        # 打印字符编号和对应的字符
        print(x,'\t',pre_characters[x])
    # 使用贪心算法计算预测结果
    out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0])*y_pred.shape[1], greedy=True)[0][0])
    # 把预测结果转化为字符串
    out = ''.join([pre_characters[x] for x in out[0]])
    # 显示图片
    plt.imshow(X_test[0])
    # 设置 title
    plt.title('pred:' + out.replace('-','') + '\ntrue: ' + captcha_text)
    # show
    plt.show()
# 生产特定验证码并进行识别
greedy('a0b1C3')




# 生产特定验证码并进行识别
# 模型训练阶段我们使用的验证码都是 3-6 位的
# 预测阶段使用 2 位长度的验证码也可以识别正确
greedy('aa')




# 模型训练阶段我们使用的验证码都是 3-6 位的
# 预测阶段使用 7 位长度的验证码也可以识别正确
# 不过由于我们的模型输入输出长度最多为 10，并且模型训练阶段，验证码最多为 6 位
# 所以如果验证码长度超过 6 的话识别的效果可能不太理想
greedy('abcdefg')


# 使用 beam search 预测结果
def beam_search(captcha_text):
    # 自定义产生一个验证码
    captcha_text = captcha_text
    # 产生验证码并归一化
    image = ImageCaptcha(width=160, height=60)
    x = np.array(image.generate_image(captcha_text)) / 255.0
    # 变成 4 维数据
    X_test = np.expand_dims(x, axis=0)
    # 用模型进行预测
    y_pred = model.predict(X_test)
    # 最好的 3 个结果
    top_paths = 3
    # 保存最好的 3 个结果
    outs = []
    for i in range(top_paths):
        labels = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1],
        greedy = False, top_paths = top_paths)[0][i])[0]
    outs.append(labels)
    # 最好的 3 个结果分别显示出来
    for out in outs:
        # 转字符串
        out = ''.join([pre_characters[x] for x in out])
        # 显示图片
        plt.imshow(X_test[0])
        # 设置 title
        plt.title('pred:' + out.replace('-','') + '\ntrue: ' + captcha_text)
        # show
        plt.show()

# 生产特定验证码并进行识别
beam_search('AbCd70')




