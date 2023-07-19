# 对不同的关系设置不同的消息传递和聚合函数
import dgl
import torch as th
import dgl.nn.pytorch as dglnn
# 先定义一个异质图
edge1 = (th.tensor([0, 1, 2]), th.tensor([2, 1, 0]))
edge2 = (th.tensor([0, 1, 2]), th.tensor([2, 1, 0]))
edge3 = (th.tensor([0, 1, 2]), th.tensor([2, 1, 0]))

graph_data = {
    ('user', 'follows', 'user') : edge1,
    ('user', 'plays', 'game') : edge2,
    ('user', 'sells', 'game') : edge3
}
g = dgl.heterograph(graph_data)

# 调用异质图卷积模块
# 声明每种关系的处理模块
mods = {
    'follows': dglnn.GraphConv(...), # 这里面的参数应该写什么呢
    'plays': dglnn.GraphConv(...),
    'sells': dglnn.SAGEConv(...)
}
conv = dglnn.HeteroGraphConv(mods=mods, aggregate='sum')

# 给所有节点声明特征
node_feat = {
    'user': th.randn(g.num_nodes('user'), 5),
    'game': th.randn(g.num_nodes('game'), 4),
    'store': th.randn(g.num_nodes('store'), 3)
}
# 调用NN模块的forward函数，返回更新后的图的节点表示
g_updated = conv(g, node_feat)
print(g_updated.keys())