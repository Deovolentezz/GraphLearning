# 内置函数和消息传递API
import dgl
import torch as th
import dgl.function as fn
# 声明一些节点
u_nodes = th.tensor([0, 1, 2, 3])
v_nodes = th.tensor([3, 2, 0, 2])
g = dgl.graph((u_nodes, v_nodes))
# 给节点添加特征，给每个节点生成一个名称为he的3维特征
g.ndata['hu'] = th.zeros(g.num_nodes(), 3)
g.ndata['hv'] = th.ones(g.num_nodes(), 3)
# 给边添加特征
g.edata['he'] = th.ones(g.num_edges(), 3)

# 消息传递函数
# 使用内置函数进行消息聚合，内置函数支持add, sub, mul, div, dot
dgl.function.u_add_v('hu', 'hv', 'he')
# 等价于下面这个用户自定义消息函数，将源节点的'hu'特征+目标节点的'hv'特征保存至边上的'he'特征
def message_func(edges):
    return {'he': edges.src['hu'] + edges.dst['hv']}

# 聚合函数，将上一步传递的消息字段'm'求和保存给一个新的字段名称'h'
dgl.function.sum('m', 'h')
# 用户自定义消息聚合函数
def reduce_func(nodes):
    return {'h': th.sum(nodes.mailbox['m'], dim=1)}

# 对图的每一条边应用消息函数，
# apply_edges的参数是一个消息函数
g.apply_edges(fn.u_add_v('hu', 'hv', 'he'))

# 更新图：通过update_all() API更新整个图的节点特征、边特征
# 将源节点特征ft与边特征a相乘得到目的节点获得的消息m，将所有消息求和更新目的节点
def update_all_example(graph):
    # 更新函数不指定
    graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
    # 对输出的节点特征进行操作
    final_ft = graph.ndata['ft'] * 2
    return final_ft


"""
编写高效的消息传递代码
"""
# 对节点特征降维减少消息维度：某些情况下必须在边上保存消息，用户需要调用apply_edges()，
# 当边上消息是高维的，会非常消耗内存
# 因此DGL建议尽量减少边上的特征维度，而这里使用的方法就是对节点特征降维来减小消息维度
# 大概意思是需要分别对源节点和目标节点降维，然后在对两者进行运算，会大大见降低空间
# 下面这一行不是很懂
node_feat_dim = 3
out_dim = 1
liner_src = th.nn.Parameter(th.FloatTensor(size=(node_feat_dim, out_dim)))
liner_dst = th.nn.Parameter(th.FloatTensor(size=(node_feat_dim, out_dim)))
# 将节点进行线性变换，降维
out_src = g.ndata['feat'] @ liner_src
out_dst = g.ndata['feat'] @ liner_dst
# 新增一个降维后的节点属性
g.srcdata.update({'out_src': out_src})
g.dstdata.update({'out_dst': out_dst})
g.apply_edges(fn.u_add_v('out_src', 'out_dst', 'out'))


# 只在图的一部分上进行消息传递
# 用户只想更新图的部分节点
nid = [0, 2, ]














