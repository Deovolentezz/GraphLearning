import dgl
import torch as th
u, v = th.tensor([0, 0, 0, 1]), th.tensor([1, 2, 3, 3])
g = dgl.graph((u, v), idtype=th.int32)
# 节点特征
# 生成节点个数行，3列的节点特征矩阵
g.ndata['node_feature'] = th.ones(g.num_nodes(), 3)

# 边特征

# 边权重