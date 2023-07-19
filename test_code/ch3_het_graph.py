# 异构图上的GraphConv模块
# 每个关系上的DGL NN模块
# 聚合来自不同关系上的结果
import dgl
import torch.nn as nn
# DGL NN模块用于处理一种关系的消息传递和聚合

class HeteroGraphConv(nn.Module):
    """
    参数mods: dict类型数据，key：关系名称，value：作用在该关系上的NN模块对象。
    aggregate: 指定如何聚合来自不同关系的结果。
    """
    def __init__(self, mods, aggregate='sum'):
        super(HeteroGraphConv, self).__init__()
        self.mods = nn.ModuleDict(mods)
        if isinstance(aggregate, str):
            # 获取聚合函数的内部函数
            self.agg_fn = get_aggregate_fn(aggregate)
        else:
            self.agg_fn = aggregate
    
    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {nty : [] for nty in g.dsttypes}
        if g.is_block:
            src_inputs = inputs
            dst_inputs = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            src_inputs = dst_inputs = inputs

        for stype, etype, dtype in g.canonical_etypes:
            rel_graph = g[stype, etype, dtype]
            if rel_graph.num_edges() == 0:
                continue
            if stype not in src_inputs or dtype not in dst_inputs:
                continue
            dstdata = self.mods[etype](
                rel_graph,
                (src_inputs[stype], dst_inputs[dtype]),
                *mod_args.get(etype, ()),
                **mod_kwargs.get(etype, {}))
            outputs[dtype].append(dstdata)
        
        rsts = {}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                rsts[nty] = self.agg_fn(alist, nty)
        return rsts