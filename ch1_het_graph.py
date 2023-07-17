# 使用DGL构建异质图
import dgl
import torch as th
# 创建一个具有3种节点和3种边的异构图
# 用三元组字典描述图
graph_data = {
    ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])), # 第一种类型的边
    ('drug', 'interacts', 'gene'): (th.tensor([0, 1]), th.tensor([2, 3])), # 第二种类型的边
    ('drug', 'treats', 'disease'): (th.tensor([1]), th.tensor([2]))
}
g = dgl.heterograph(graph_data)
# 通过上述的图构建，会得到drug类型节点3个，gene类型节点4个，disease节点3个，边5条
# 同构图就是节点类型和边类型只有一个，二分图是边类型只有一个
# 上述的一个三元组字典即为一个metagraph，就是图的模式，指定节点集和节点之间的边类型约束。

# 获取特定类型信息时，需要指定具体的节点和边类型
drug_nodes_num = g.num_nodes('drug')
gene_nodes_num = g.num_nodes('gene')
disease_nodes_num = g.num_nodes('disease')
print(drug_nodes_num, gene_nodes_num, disease_nodes_num)

# 给某一类节点初始化特征
# 给所有的drug节点初始化特征，特征长度为3
g.nodes['drug'].data['hv'] = th.ones(g.num_nodes('drug'), 2)
# drug节点特征向量
print(g.nodes['drug'].data['hv'])
# 给所有的interacts边初始化特征，给所有interacts边生成一个长度为1的特征
g.edges[('drug', 'interacts', 'drug')].data['weight'] = th.zeros(g.num_edges(('drug', 'interacts', 'drug')), 1)
print(g.edges[('drug', 'interacts', 'drug')].data['weight'])


# 注意：
# 通过g.edge[(源类型，关系，目的类型)]获取特定类型的边，而不能像g.nodes['']一样只通过中间的关系类型字段获取特定的边

# 从磁盘加载异构图，每种类型的节点单独形成一个csv文件，每种边也单独形成一个csv文件

# 边类型子图：通过保留特定类型的边，创建异构图的子图
eg = dgl.edge_type_subgraph(g, [('drug', 'interacts', 'drug'), ('drug', 'treats', 'disease')])


# 若想要对不同类型的边进行相同的操作，则可以先抽取子图，再将子图转变为同质图


