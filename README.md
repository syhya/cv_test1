# 卷积神经网络实现手写数字识别（Convolutional neural network for handwritten digit recognition）

#### 介绍
随着智能设备的发展，手写输入为人们的交流带来了极大的方便，而本项目中所进行的手写数字识别，则是手写输入的一部分，通过对手写数字识别进行讨论，掌握卷积神经网络在图片识别中的超强应用以及对手写输入产生入门级的认识。本项目利用了Keras自带的训练和测试数据集-MNIST数据集。

#### 数据集来源和加载
Keras库提供了一种加载MNIST数据的简单方法。即数据集以mnist.pkl.gz(15M)文件形式自动下载到用户的目录中.由于Keras自带了训练和测试数据集，数据格式也都已经整理完毕，因此加载MNIST数据集就是调用mnist.load_data()函数:

```
(XTrain, yTrain), (XTest, yTest) = mnist.load_data()

```

#### 预处理

自带数据集尺寸已归一化并保证图像居中。每张图片均为28×28=784个像素。每个像素都有与其相关联的单个像素值，用来指示该像素的明暗度，值为0~255之间（包括0和255），数字越大图像越暗。因此预处理过程只需要搭建Keras模块，并且确保训练集和测试集的数据和模块的参数相吻合即可。处理过程如下：



```
(XTrain, yTrain), (XTest, yTest) = mnist.load_data()
plt.subplot(221)
plt.imshow(XTrain[1], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(XTrain[2], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(XTrain[3], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(XTrain[4], cmap=plt.get_cmap('gray'))
plt.show()
```



![示例图](https://images.gitee.com/uploads/images/2021/1013/235235_1748b1a0_7659950.png "屏幕截图.png")

#### 模型建立

   1.用最简单的架构实现CNN的第一层。对于序贯模型，对层进行堆叠，并在第一层，也就是卷积层Conv2D（）中指定图像的输入维度、特征映射数量、输入形状和激活函数（rule)，然后添加内核维度为2×2的最大池化层。


```
model.add(Conv2D(40, kernel_size=5, padding="same", input_shape=(28, 28, 1), activation='relu'))
model.add(Conv2D(50, kernel_size=5, padding="valid", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

```


   2.添加一个平展层，用于接收CNN的输出并将其平展，再将平展后的数据作为稠密层的输入，进而将其传递到输出层。在输出层使用softmax输出多类分类的预测概率。


```
model.add(Flatten())
model.add(Dense(100))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation("softmax"))
```


![完整CNN的流程图](https://images.gitee.com/uploads/images/2021/1013/235718_6a5aefc0_7659950.png "屏幕截图.png")

随着网络的加深，CNN会逐渐减少层的维数，并增加特征映射的数量，从而在检测更多特征的同时降低计算成本。

#### 参数设置和调整

（1）为了提高算法的效率和收敛性，基于最大像素255对数据进行归一化处理，将所有的像素除以255以将输入变为0~1直接的数值。

```
XTrain = XTrain / 255
XTest = XTest / 255
```


（2）根据数据选择合适的算法并得到所需的准确率指标。如果数据与类别的分布比较均衡，则简单的使用准确率；如果数据与类别分布不均衡，那么将无法使用准确率，因为其结果会产生误导，这时候就是用另一种指标。从下图中可以看到，数据集由60000个训练样本组成，其中每个样本都是28×28的图像。由于类的分布是均衡的，因此我们可以使用准确率作为度量标准。


Number of training examples = 60000
Number of classes = 10
Dimension of images = 28 x 28  
The number of occurrences of each class in the dataset = {0: 5923, 1: 6742, 2: 5958, 3: 6131, 4: 5842, 5: 5421, 6: 5918, 7: 6265, 8: 5851, 9: 5949}   

（3）重塑数据样本或图像，以使他们适合用卷积神经网络进行训练。在Keras库中，各层使用图像像素的维度为(pixels)(width)(height)。而在MNIST中，像素值就是灰度值，所以像素的维度为1.以独热编码的形式输出结果，由于输出层需要10个节点，因此把目标数字0~9做成独热编码的形式。


```
XTrain = XTrain.reshape(XTrain.shape[0], 28, 28, 1).astype('float32')
XTest = XTest.reshape(XTest.shape[0], 28, 28, 1).astype('float32')
yTrain = np_utils.to_categorical(yTrain)
yTest = np_utils.to_categorical(yTest)
```


#### 训练结果

通过训练48000个样本并在12000个样本上进行验证获得输出。由下图输出可以看到，本网络的架构的准确率可以达到99%以上。

![准确率图](https://images.gitee.com/uploads/images/2021/1014/001017_d6ebe88d_7659950.png "屏幕截图.png")

#### 小结

本次实验我通过使用卷积神经网络及进行手写字体的分类，从而对卷积神经网络、深度学习有了更深的理解和应用。
