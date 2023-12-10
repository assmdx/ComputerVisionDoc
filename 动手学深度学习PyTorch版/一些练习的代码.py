# 5.1.4 练习1
class MySequential1(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[idx] = module
    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X
net = MySequential1(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)
# print(net, '\n', net.state_dict()) # 无法像_modules一样使用net.state_dict()方便的访问模型的网络结构和参数

# 5.1.4 练习2
import torch
from torch import nn
from torch.nn import functional as F
class Parallel(nn.Module):
    def __init__(self, net1, net2):
        super().__init__()
        self.net1 = net1
        self.net2 = net2
    def forward(self, X):
        x1 = self.net1(X)
        x2 = self.net2(X)
        return torch.cat((x1, x2), 1)
#         return x2
X = torch.rand(2, 20)
net = Parallel(nn.Linear(20, 20), nn.Linear(20, 20))
output = net(X)
X.size(), output.size()

# 5.1.4 练习3
def create_network(num, input_size, hidden_size, output_size):
    linear_network = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, input_size))
    instances = [linear_network for _ in range(num)]
    net = nn.Sequential(*instances)
    net.add_module("output", nn.Linear(input_size, output_size))
    return net

# 初始化权重的示例，正态分布初始化权重，bias初始化为0
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0,std=0.01)
        nn.init.zeros_(m.bias)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())
net.apply(init_normal)

# 遍历神经网络的技巧
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()][0])
net.apply(my_init)

# 查看神经网络的参数示例代码
net[2].weight.data[0]

