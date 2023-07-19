# 处理整图分类数据集
import dgl
import numpy as np
import torch
from rgcn import RGCN
import torch.nn.functional as F

"""
内置样例数据：有user item两种节点，
有follow followed-by click clicked-by dislike disliked-by
6种关系
"""
n_users = 1000
n_items = 500
n_follows = 3000 # 存在follow关系的边有3000条
n_clicks = 5000 # 存在click关系的边有5000条
n_dislikes = 500
n_hetero_features = 10 # 这是啥啊
n_user_classes = 5 # 这是啥
n_max_clicks = 10
dataset = dgl.data.CiteseerGraphDataset()
graph = dataset[0]

# 生成从0开始到n_user为止共计n_follows整数（因为在follow关系中
# 一共就n_follow条边，n_follow条边对应n_follow个起点和终点
# 这里可以替换为实际场景中的节点类型
follow_src = np.random.randint(0, n_users, n_follows)
follow_dst = np.random.randint(0, n_users, n_follows)
click_src = np.random.randint(0, n_users, n_clicks)
click_dst = np.random.randint(0, n_items, n_clicks)
dislike_src = np.random.randint(0, n_users, n_dislikes)
dislike_dst = np.random.randint(0, n_items, n_dislikes)

# 构建异质图
hetero_graph = dgl.heterograph({
    ('user', 'follow', 'user'): (follow_src, follow_dst),
    ('user', 'followed-by', 'user'): (follow_dst, follow_src),
    ('user', 'click', 'item'): (click_src, click_dst),
    ('item', 'clicked-by', 'user'): (click_dst, click_src),
    ('user', 'dislike', 'item'): (dislike_src, dislike_dst),
    ('item', 'disliked-by', 'user'): (dislike_dst, dislike_src)
})

# 给异质图的节点附上特征
# user节点和item节点的特征维度相同
hetero_graph.nodes['user'].data['feature'] = torch.randn(n_users, n_hetero_features)
hetero_graph.nodes['item'].data['feature'] = torch.randn(n_items, n_hetero_features)
# 给节点添加标签
hetero_graph.nodes['user'].data['label'] = torch.randint(0, n_user_classes, (n_users,))
# 给边添加标签
hetero_graph.edges['click'].data['label'] = torch.randint(1, n_max_clicks, (n_clicks,)).float()

# 在user类型的节点和click类型的边上随机生成训练集的掩码
hetero_graph.nodes['user'].data['train_mask'] = torch.zeros(n_users, dtype=torch.bool).bernoulli(0.6)
hetero_graph.edges['click'].data['train_mask'] = torch.zeros(n_clicks, dtype=torch.bool).bernoulli(0.6)

# 图节点分类/回归
"""
1. 图数据中的训练、验证和测试集中的每个节点都有一个类别/正确的标注（这是一定的吗，
2. 为了对节点进行分类，图神经网络执行消息传递机制，利用节点自身的特征和其邻节点及边的
特征计算节点的隐藏表示，而且消息传递可以重复多轮以利用更大范围的邻居信息
"""
model = RGCN(n_hetero_features, 20, n_user_classes, hetero_graph.etypes)

# 获取节点特征及标签
user_feats = hetero_graph.nodes['user'].data['feature']
item_feats = hetero_graph.nodes['item'].data['feature']
labels = hetero_graph.nodes['user'].data['label']
train_mask = hetero_graph.nodes['user'].data['train_mask']

# 进行前向传播计算
node_features = {'user': user_feats, 'item': item_feats}
h_dict = model(hetero_graph, {'user': user_feats, 'item': item_feats})
h_user = h_dict['user']
h_item = h_dict['item']

opt = torch.optim.Adam(model.parameters())

for epoch in range(5):
    model.train()
    # 使用所有节点的特征进行前向传播计算，并提取输出的user节点嵌入
    logits = model(hetero_graph, node_features)['user']
    # 计算损失值
    loss = F.cross_entropy(logits[train_mask], labels[train_mask])
    # 计算验证集的准确度。在本例中省略。
    # 进行反向传播计算
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(loss.item())

    # 如果需要的话，保存训练好的模型。本例中省略。