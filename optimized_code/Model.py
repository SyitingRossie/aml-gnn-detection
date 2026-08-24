from torch_geometric.nn import GATv2Conv
class ImprovedGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, edge_dim=None):
        super().__init__()

        self.conv1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,  
            edge_dim=edge_dim,
            dropout=0.3, 
            fill_value=0,
            add_self_loops=True   
        )

        # 验证第一层输出后的维度计算：hidden_channels * heads = 16 * 8 = 128
        conv1_output_dim = hidden_channels * heads

        # 批归一化层：紧跟在第一层卷积输出之后
        self.bn1 = nn.BatchNorm1d(conv1_output_dim)

        self.conv2 = GATv2Conv(
            in_channels=conv1_output_dim,
            out_channels=out_channels,
            heads=1,
            concat=False,       
            edge_dim=edge_dim,
            dropout=0.3,
            fill_value=0,
            add_self_loops=True
        )

    def forward(self, x, edge_index, edge_attr=None,return_embedding=False):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        if return_embedding:
            return x
        x = self.conv2(x, edge_index,edge_attr=edge_attr)

        return x.squeeze(-1)           #将[batch_size, 1] 压平为 [batch_size]，匹配损失函数维度


# 8.模型构建[ORIGINAL] （参考）
# class GAT(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, heads, edge_dim):
#         super().__init__()
#         self.conv1 = GATConv(in_channels=in_channels, out_channels=hidden_channels, heads=heads, dropout=0.3,
#                              add_self_loops=True,
#                              edge_dim=edge_dim)  # 卷积加入自生原始特征！！！#dropout从0.6改成0.3 ,原作者没有加入边特征！！！！！！！！！！！！1
#         self.conv2 = GATConv(hidden_channels * heads, int(hidden_channels / 4), heads=1,
#                              dropout=0.3,
#                              edge_dim=edge_dim)  # concat=False，这句没加，因为concat只有在heads>1时才有区别。concat=True是默认的，横向拼接所有头的特征，如果是False，则是取平均，不拼接。
#         self.lin = Linear(int(hidden_channels / 4), out_channels)  # 从4到1
#         # self.sigmoid = nn.Sigmoid()

#     def forward(self, x, edge_index, edge_attr):  # edge_index是二维张量，第一行是from的账户index，第二行是to的账户index。   输入x：是节点特征
#         x = F.dropout(x, p=0.3, training=self.training)
#         x = F.elu(self.conv1(x, edge_index, edge_attr))
#         x = F.dropout(x, p=0.3, training=self.training)
#         x = F.elu(self.conv2(x, edge_index, edge_attr))
#         logits = self.lin(x)
#         return logits.squeeze(dim=1)

