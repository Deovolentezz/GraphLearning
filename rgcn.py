# rgcn模型
# 首先对每种边类型进行单独的图卷积运算
# 然后将每种边类型上的消息聚合结果再相加
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F

class RGCN(nn.Module):
    # 这里的hid_feats初始时是什么，是用来接收第一层图神经网络将in_feats embedding之后的特征吗
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        # 实例化HeteroGraphConv
        # in_feats是输入特征维度，而不是输入特征
        # out_feats是输出特征维度，而不是
        # aggregate是聚合函数函数的类型

        # 第一层图神经网络
        # HeteroGraphConv传入的两个参数：一个是每种关系的消息传递机制的字典
        # 另一个是所有关系消息聚合的方式
        self.conv1 = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GraphConv(in_feats, hid_feats)
                for rel in rel_names
            }, aggregate='sum'
        )

        # 第二层图神经网络
        # 将上一层网络的输出作为输入
        # 声明了这个类并Init，但没有调用forward函数
        self.conv2 = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GraphConv(hid_feats, out_feats)
                for rel in rel_names
            }, aggregate='sum'
        )
    
    def forward(self, graph, inputs):
        # Inputs为节点特征字典
        # 返回的h也是每类节点的新特征
        h = self.conv1(graph, inputs)
        # 将新特征激活一下再传入到第二层
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h
