import dgl
import torch as th
u, v = th.tensor([0, 0, 0, 1]), th.tensor([1, 2, 3, 3])
g = dgl.graph((u, v), idtype=th.int32)
# 节点特征
# 生成节点个数行，3列的节点特征矩阵
g.ndata['node_feature'] = th.ones(g.num_nodes(), 3)

# 边特征，给每一个边赋一个数值特征
g.edata['edge_weight'] = th.ones(g.num_edges(), dtype=th.float32)

# 获取某一个节点/某一条边的特征
n_feature = g.ndata['node_feature'][0] # 获得第0个节点的'node_featue'特征
e_feature = g.edata['edge_weight'][0]

print(n_feature)
print(e_feature)
# 获取多个节点/边的特征
n_features = g.ndata['node_feature'][th.tensor([0, 2])] # 获得第0个和第2个节点特征
print(n_features)

# 可以通过csv导入节点数据和边数据
# 可以通过dgl.save_graphs()和dgl.load_graphs()存储和加载图数据。
# 下面这个graph label不能是字符串类型的label
graph_labels = {"glabel":th.tensor([0])}
# dgl.save_graphs('data/sample_graph.bin', [g], graph_labels)