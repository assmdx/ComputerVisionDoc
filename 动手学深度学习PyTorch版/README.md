> 笔记基于《动手学深度学习PyTorch版》

[torch.normal函数](./torch.normal.md)

定义一个线性回归的模型,第一个参数2是输入特性的形状，第二个参数指定输出形状，输出形状为标量，则为1。

nn是神经网络的意思。

```
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))
```
一个pytorch线性神经网络示例：
```
import numpy as np
import torch
import d2l
from torch.utils import data

# 预定义一组答案参数
true_w = torch.tensor([2, -3.4]) # 2 * 1
true_b = 4.2

# 随机生成数据，加上噪声，噪声符合正态分布
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w))) # 1000 * 2
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))
features, labels = synthetic_data(true_w, true_b, 1000)

# 训练的时候取batch批量数据
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
batch_size = 10
data_iter = load_array((features, labels), batch_size)
next(iter(data_iter))

# 定义线性回归神经网络
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))

# 初始化神经网络参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 定义神经网络的损失函数
loss = nn.MSELoss() # L2范数


# 定义优化算法（这里用的小批量随机梯度下降算法）：reference: https://blog.csdn.net/weixin_39228381/article/details/108310520
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 定义训练几轮，每轮把所有数据训练一遍
num_epochs = 10
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad() # trainer.zero_grad()用于将记录梯度的矩阵清零，网络进行训练的过程中，我们会存储两个矩阵：分别是params矩阵用于存储权重参数；以及params.grad用于存储梯度参数，reference: https://blog.csdn.net/qq_43369406/article/details/129740629
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l: f}')

# 打印训练结果与答案的偏差
w = net[0].weight.data
print('w的误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的误差：', true_b - b)
```

pytorch的其他优化器算法：（todo_rzc）: https://zhuanlan.zhihu.com/p/64885176

超参数：
1. 学习率
    1. 通常的，在训练网络的前期过程中，会选取一个相对较大的学习率以加快网络的收敛速度。而随着迭代优化的次数增多，逐步减小学习率，以保证最终收敛至全局最优解，而不是在其附近震荡或爆炸。几种常用的学习率衰减方法（todo_rzc: 学习一下）：
        - 分段常数衰减
        - 指数衰减
        - 自然指数衰减
        - 多项式衰减
        - 间隔衰减
        - 多间隔衰减
        - 逆时间衰减
        - Lambda衰减
        - 余弦衰减
        - 诺姆衰减
        - loss自适应衰减
        - 线性学习率热身
2. epoch
    1. epoch调优的技巧，reference: https://www.27ka.cn/39160.html
        1. 概念： 在深度学习中，一个epoch表示所有训练样本都被送入网络中进行了一次训练
        2. 设置合理的batch size：batch size是指每次迭代所选取的样本数，**对于大型数据集，我们通常会使用小batch size来加速训练。而对于小型数据集，通常使用大batch size能够更好地利用硬件加速。**调整batch size的大小可以帮助我们更好地利用GPU并行计算的优势，从而加速训练过程。
        3. 合理使用学习率调整策略：学习率是指模型在权重更新时所采用的参数更新速度。合理的学习率可以使模型更快地收敛，并提高模型的准确率。当模型的学习率过高或过低时，都可能导致模型的训练效果无法得到充分发挥。常用的学习率调整策略有：
            - 按步数调整
            - 按指数调整
            - 余弦退火调整等。
        4. 使用合适的正则化方法（todo_rzc: 什么是正则化：reference: https://blog.csdn.net/qq_43966129/article/details/123029320）：防止过拟合。在进行epoch调优时，我们通常使用L1或L2正则化来限制网络的权重，避免过拟合现象的发生。
        5. 增加数据集的多样性
        6. 合理选择损失函数：损失函数是评价模型优劣的一个重要指标，选用合适的损失函数可以使模型收敛更快并提高准确率。在epoch调优过程中，我们需要根据实际情况选择合适的损失函数。常用的损失函数：
            - 均方误差（MSE）
            - 交叉熵（Cross Entropy）
            - 对数损失
3. batch_size
4. 其他：初始参数？（todo_rzc: 初始参数可能不算超参数？）





