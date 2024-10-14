from torch.nn import TransformerEncoder, TransformerEncoderLayer
from layers import *
from torch_geometric.nn import GCNConv
from sklearn.neighbors import NearestNeighbors


class FuseTransEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, nhead):
        super(FuseTransEncoder, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=False)
        self.transformerEncoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.d_model = hidden_size
        self.sigal_d = int(self.d_model/2)
        
        self.res_img = nn.Linear(self.sigal_d, self.sigal_d)
        self.res_txt = nn.Linear(self.sigal_d, self.sigal_d)

    def forward(self, tokens):
        encoder_X = self.transformerEncoder(tokens)
        encoder_X = encoder_X.float()
        encoder_X_r = encoder_X.reshape(-1, self.d_model)
        encoder_X_r = normalize(encoder_X_r, p=2, dim=1)  # p 指规范化的类型，dim 表示在哪个维度上进行规范化

        img, txt = encoder_X_r[:, :self.sigal_d], encoder_X_r[:, self.sigal_d:]
        img = img + self.res_img(img)
        txt = txt + self.res_txt(txt)
        
        return img, txt



class PGCH(nn.Module):
    def __init__(self, bits=128, classes=24): # 模型的哈希码位数
        super(PGCH, self).__init__()
        self.bits = bits
        self.act = nn.Tanh()
        self.mde = MDE(hidden_dim=[512, 2048, 1024, self.bits], act=nn.Tanh())

        self.classify = nn.Linear(self.bits, classes)  # 分类器

        self.discriminator = nn.Sequential(
            nn.Linear(self.bits, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, txt):
        img = img.clone().detach().requires_grad_(True).to(torch.float)
        txt = txt.clone().detach().requires_grad_(True).to(torch.float)
        img_common, txt_common = img, txt
        hash1 = self.mde(img_common)  # 计算图像哈希值
        hash2 = self.mde(txt_common)  # 计算文本哈希值

        D1 = self.discriminator(hash1)  # 对图像哈希值进行判别
        D2 = self.discriminator(hash2)  # 对文本哈希值进行判别
        return img_common, txt_common, hash1, hash2, self.classify(hash1), self.classify(hash2), D1, D2


class GCN(nn.Module):
    def __init__(self, num_feature, num_class, hidden_size, dropout=0.5, activation="relu"):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_feature, hidden_size)
        self.conv2 = GCNConv(hidden_size, num_class)

        self.dropout = dropout
        assert activation in ["relu", "leaky_relu", "elu"]
        self.activation = getattr(F, activation)

    def forward(self, feature, adj):
        x1 = self.activation(self.conv1(feature, adj))
        x1 = F.dropout(x1, p=self.dropout, training=self.training) # 在训练时才会执行dropout
        x2 = self.conv2(x1, adj)
        return x1, F.log_softmax(x2, dim=1)






class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, dim_feedforward, dropout=0.5, k=5):
        super(Encoder, self).__init__()
        self.linear_dim = nn.Linear(output_dim, input_dim)
        self.fusion_layer = nn.Linear(input_dim * 2, input_dim)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, input_dim),
        )

        self.norm1 = nn.BatchNorm1d(input_dim)
        self.norm2 = nn.BatchNorm1d(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.Tanh(), )

        self.k = k  # KNN 的最近邻数量
        
    def forward(self, src, anchor_2):
        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='auto').fit(anchor_2.cpu().detach().numpy())
        distances, indices = nbrs.kneighbors(src.cpu().detach().numpy())

        indices = torch.tensor(indices, dtype=torch.long).to(anchor_2.device)

        expanded_anchor_2 = anchor_2.unsqueeze(0).expand(indices.size(0), -1, -1)  # 扩展维度
        anchor2_neighbors = torch.gather(expanded_anchor_2, 1, indices.unsqueeze(-1).expand(-1, -1, anchor_2.size(-1)))
        anchor2_neighbors = anchor2_neighbors.mean(dim=1)

        anchor2_mapped = self.linear_dim(anchor2_neighbors)

        combined_input = torch.cat((src, anchor2_mapped), dim=1)
        combined_features = self.fusion_layer(combined_input)

        combined_features = self.norm1(combined_features)
        transformed_features = self.encoder(combined_features)
        combined_features = combined_features + self.dropout2(transformed_features)
        combined_features = self.norm2(combined_features)

        src = self.decoder(combined_features)

        return src
