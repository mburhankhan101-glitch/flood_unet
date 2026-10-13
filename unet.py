"""A U-Net for 2-class segmentation, four levels deep (input sizes must be multiples of 16)."""
import torch
from torch import nn


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Encoder: 4 times (two convolutions, then halve). Decoder: double back up, joining the encoder's
    output at the same size (skip connection). At the bottom one cell covers 16 x 16 input pixels."""

    def __init__(self, cin=2, n_out=2, base=16, dropout=0.2):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8, base * 16]
        self.down = nn.ModuleList([block(cin, ch[0])] + [block(ch[i], ch[i + 1]) for i in range(4)])
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(dropout)
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(4))])
        self.dec = nn.ModuleList([block(ch[i] * 2, ch[i]) for i in reversed(range(4))])
        self.out = nn.Conv2d(ch[0], n_out, 1)

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.down):
            x = d(x if i == 0 else self.pool(x))
            skips.append(x)
        x = self.drop(skips.pop())                       # the bottom: the widest view
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], dim=1))
        return self.out(x)
