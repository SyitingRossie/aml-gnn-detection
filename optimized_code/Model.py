# 优化后模型
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
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
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch_geometric.transforms as T
# from torch_geometric.nn import GATConv, Linear

# class GAT(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, heads):
#         super().__init__()
#         self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
#         self.conv2 = GATConv(hidden_channels * heads, int(hidden_channels/4), heads=1, concat=False, dropout=0.6)
#         self.lin = Linear(int(hidden_channels/4), out_channels)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, edge_index, edge_attr):
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = F.elu(self.conv1(x, edge_index, edge_attr))
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = F.elu(self.conv2(x, edge_index, edge_attr))
#         x = self.lin(x)
#         x = self.sigmoid(x)
        
#         return x

