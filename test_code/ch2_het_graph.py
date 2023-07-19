# 在异质图上进行消息传递
# 异质图上的消息传递分为两部分：1）计算和聚合同种关系的消息；2）对每个节点聚合来自不同关系的消息
import dgl.function as fn
import torch as th
import dgl
def update_het_graph():
    graph_data = {
        ('drug', 'interact', 'gene'): (th.tensor([0, 1, 2]), th.tensor([1, 2, 3])),
        ('drug', 'treat', 'disease'): (th.tensor([0, 1, 2]), th.tensor([1, 2, 3]))
    }
    G = dgl.heterograph(graph_data)
    # 给节点添加特征
    G.nodes['drug'].data['time'] = th.ones(G.num_nodes('drug'), 3)
    G.nodes['gene'].data['len'] = th.ones(G.num_nodes('gene'), 3)
    G.nodes['disease'].data['type'] = th.ones(G.num_nodes('disease'), 3)


    # 遍历异质图中的每种关系，设置每种关系的消息传递和聚合函数
    # 定义multi_update_all()需要传入的字典
    funcs = {}
    for c_etype in G.canonical_etypes:
        # 聚合一种类型的消息
        srctype, etype, dsttype = c_etype
        # 给源节点一个权重，源节点在etype关系下的权重
        G.nodes[srctype].data['weight_%s' % etype] = th.randn(G.num_edges())
        funcs[etype] = (fn.copy_u('weight_%s' % etype, 'm'), fn.mean('m', 'h'))
    # 将每种类型聚合的消息相加
    G.multi_update_all(funcs, 'sum')
    # 返回每种类型节点更新后的特征向量（对应于初始给节点添加特征时，G.nodes[ntype].data['ft']
    return {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes}

