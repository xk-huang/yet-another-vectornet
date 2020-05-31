# %%
from torch_geometric.data import Data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import MessagePassing

# %%


class HGNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_subgraph_layres=3, num_global_graph_layer=1, subgraph_width=64, global_graph_width=64):
        super(HGNN, self).__init__()
        self.subgraph = SubGraph(
            in_channels, num_subgraph_layres, subgraph_width)
        self.traj_pred_mlp = nn.Sequential(
            nn.Linear(in_channels * (2 ** num_subgraph_layres),
                      global_graph_width),
            nn.LayerNorm(global_graph_width),
            nn.ReLU(),
            nn.Linear(global_graph_width, out_channels)
        )

    def forward(self, data):
        sub_data = data
        # Not implement data spliting by polyline_id and global interaction,
        polyline_feature = self.subgraph(sub_data)
        # print(polyline_feature.shape)
        # print(self.traj_pred_mlp)
        out = self.traj_pred_mlp(polyline_feature)

        return out

# %%


class SubGraph(nn.Module):
    def __init__(self, in_channels, num_subgraph_layres=3, hidden_unit=64):
        super(SubGraph, self).__init__()
        self.num_subgraph_layres = num_subgraph_layres
        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layres):
            self.layer_seq.add_module(
                f'glp_{i}', GraphLayerProp(in_channels, hidden_unit))
            in_channels *= 2

    def forward(self, sub_data):
        """
        polyline vector set in torch_geometric.data.Data format
        args:
            sub_data: torch_geometric.data.Data(x, edge_index)
        """
        x, edge_index = sub_data.x, sub_data.edge_index
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, GraphLayerProp):
                x = layer(x, edge_index)
        node_feature, _ = torch.max(x, dim=0)
        # l2 noramlize node_feature before feed it to global graph
        node_feature = node_feature / node_feature.norm(dim=0)
        return node_feature

# %%


class GraphLayerProp(MessagePassing):
    def __init__(self, in_channels, hidden_unit=64):
        super(GraphLayerProp, self).__init__(
            aggr='max')  # MaxPooling aggragation
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU(),
            nn.Linear(hidden_unit, in_channels)
        )

    def forward(self, x, edge_index):
        # print(x.shape)
        x = self.mlp(x)
        aggr_x = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
        # print(x)
        # print(aggr_x)
        assert x.shape == aggr_x.shape, "aggr shape not the same."

        return torch.cat([x, aggr_x], dim=1)

    def message(self, x_j):
        return x_j

    def update(self, x):
        return x


# %%
if __name__ == "__main__":
    in_channels, out_channels = 3, 4
    epochs = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HGNN(in_channels, out_channels).to(device)
    edge_index = torch.tensor(
        [[1, 2, 0, 2, 0, 1],
            [0, 0, 1, 1, 2, 2]], dtype=torch.long)
    x = torch.tensor([[3, 1, 2], [2, 3, 1], [1, 2, 3]], dtype=torch.float)
    y = torch.tensor([1, 2, 3, 4], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, y=y)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        # print(out.shape, y.shape)
        loss = F.mse_loss(out, y)
        loss.backward()
        optimizer.step()
        print(f"loss {loss.item():.3f}")
    model.eval()
    with torch.no_grad():
        out = model(data)
        loss = F.mse_loss(out, y)
        print(f"loss {loss.item():.3f}")
        print(out)
        print(y)


# %%
