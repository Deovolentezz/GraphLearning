# DGL NN模块的构造函数
import torch.nn as nn
from dgl.utils import expand_as_pair
import dgl.function as fn
import torch.nn.functional as F
from dgl.utils import check_eq_shape

class SAGEConv(nn.Moudle):
    # 构造函数
    def __init__(self, in_feats, out_feats, aggregator_type, bias=True,\
                 norm=None, activation=None):
        # 输入维度
        super(SAGEConv, self).__init__()
        # 图神经网络，输入维度分为：源节点特征维度，目标节点特征维度
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        # 输出维度，输出的是每个节点经过嵌入后特征对吗
        self._out_feats = out_feats
        # 聚合类型，决定了如何聚合不同边上的信息，常用聚合类型:mean;sum;max;min;lstm
        self._aggre_type = aggregator_type
        # 特征归一化参数，可以是l2归一化等
        self.norm = norm
        self.activation = activation

        if aggregator_type not in ['mean', 'pool', 'lstm', 'gcn']:
            raise KeyError('Aggregator type {} not supported.'.format(aggregator_type))
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)
        if aggregator_type in ['mean', 'pool', 'lstm']:
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=bias)
        self.reset_parameters()

        

    # 注册参数和子模块，子模块是干啥的
    def reset_parameters(self):
        # 重新初始化可学习的参数，这些参数是啥意思啊
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == 'lstm':
            self.lstm.reset_parameters()
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)

        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
    
    # 编写DGL NN模块的forward函数
    # 在NN模块中forward()函数执行了实际的消息传递和计算
    # 与传统的Pytorch NN模块相比，增加了一个dgl.DGLGraph参数
    # forword函数的3项操作：检测输入的图对象是否符合规范；消息传递和聚合；聚合后更新特征
    def forward(self, graph, feat):
        with graph.local_scope():
            # 指定图类型，根据图类型扩展输入特征
            feat_src, feat_dst = expand_as_pair(feat, graph)
            # 消息的传递和聚合
            if self._aggre_type == 'mean':
                graph.srcdata['h'] = feat_src
                graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))
                h_neigh = graph.dstdata['neigh']
            elif self._aggre_type == 'gcn':
                check_eq_shape(feat)
                graph.srcdata['h'] = feat_src
                graph.dstdata['h'] = feat_dst
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'neigh'))
                # 除以入度
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
            elif self._aggre_type == 'pool':
                graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
                graph.update_all(fn.copy_u('h', 'm'), fn.max('m', 'neigh'))
                h_neigh = graph.dstdata['neigh']
            else:
                raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

            # GraphSAGE中gcn聚合不需要fc_self
            if self._aggre_type == 'gcn':
                rst = self.fc_neigh(h_neigh)
            else:
                # 这里fc_self原本的参数是h_self，猜测是自己原始的特征
                # 但在上下文中没有找到h_self的定义，我替换成了feat_dst
                rst = self.fc_self(feat_dst) + self.fc_neigh(h_neigh)
            
            # 激活函数，根据构造函数中的参数选择对节点进行特征更新
            if self.activation is not None:
                rst = self.activation(rst)
            # 归一化
            if self.norm is not None:
                rst = self.norm(rst)
            return rst
