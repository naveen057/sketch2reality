import torch
import torch.nn as nn


class SketchCNN(nn.Module):
    """Small CNN for 28x28 sketches."""

    def __init__(self, num_classes: int):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))  # 28->14->7->3
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 3 * 3, 256), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class TinyViT(nn.Module):
    """Minimal Vision Transformer (patch 4x4 -> 49 tokens + CLS)."""

    def __init__(self, num_classes, img_size=28, patch=4, dim=128, depth=6, heads=4, mlp=256, dropout=0.1):
        super().__init__()
        n_tokens = (img_size // patch) ** 2
        self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            dim, heads, mlp, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1) + self.pos
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0]))


class CGANGenerator(nn.Module):
    """Class-conditional generator: (noise, label) -> 28x28 sketch in [-1, 1]."""

    def __init__(self, num_classes, z_dim=100, emb=50):
        super().__init__()
        self.z_dim = z_dim
        self.embed = nn.Embedding(num_classes, emb)
        self.fc = nn.Sequential(nn.Linear(z_dim + emb, 128 * 7 * 7), nn.BatchNorm1d(128 * 7 * 7), nn.ReLU())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),   # 14
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh(),                          # 28
        )

    def forward(self, z, y):
        h = self.fc(torch.cat([z, self.embed(y)], dim=1)).view(-1, 128, 7, 7)
        return self.net(h)


class CGANDiscriminator(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.embed = nn.Embedding(num_classes, 28 * 28)
        self.net = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Flatten(), nn.Linear(128 * 7 * 7, 1),
        )

    def forward(self, x, y):
        label_map = self.embed(y).view(-1, 1, 28, 28)
        return self.net(torch.cat([x, label_map], dim=1))
