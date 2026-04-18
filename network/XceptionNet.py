import torch
import torch.nn as nn
import torch.nn.functional as F


# NOTE:
# This Xception implementation follows the FaceForensics/DeepfakeBench variant.
# It is compatible with DeepfakeBench's "xception_best.pth" detector checkpoint.


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super().__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = in_filters

        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for _ in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class XceptionNet(nn.Module):
    """
    Xception (FaceForensics/DeepfakeBench variant).
    Returns (logits, features) where features is the last conv feature map.
    """

    def __init__(self, num_classes=2, mode="original", inc=3, dropout=False):
        super().__init__()
        self.num_classes = num_classes
        self.mode = mode

        # Entry flow
        self.conv1 = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        # Middle flow
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        final_channel = 2048
        if self.mode == "adjust_channel_iid":
            final_channel = 512
            self.mode = "adjust_channel"

        if dropout:
            self.last_linear = nn.Sequential(
                nn.Dropout(p=float(dropout)),
                nn.Linear(final_channel, self.num_classes),
            )
        else:
            self.last_linear = nn.Linear(final_channel, self.num_classes)

        self.adjust_channel = nn.Sequential(
            nn.Conv2d(2048, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
        )

    def fea_part1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

    def fea_part2(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def fea_part3(self, x):
        if self.mode == "shallow_xception":
            return x
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        return x

    def fea_part4(self, x):
        if self.mode == "shallow_xception":
            x = self.block12(x)
        else:
            x = self.block8(x)
            x = self.block9(x)
            x = self.block10(x)
            x = self.block11(x)
            x = self.block12(x)
        return x

    def fea_part5(self, x):
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def features(self, x):
        x = self.fea_part1(x)
        x = self.fea_part2(x)
        x = self.fea_part3(x)
        x = self.fea_part4(x)
        x = self.fea_part5(x)
        if self.mode == "adjust_channel":
            x = self.adjust_channel(x)
        return x

    def classifier(self, features, id_feat=None):
        if self.mode == "adjust_channel":
            x = features
        else:
            x = self.relu(features)
        if x.dim() == 4:
            x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        self.last_emb = x
        if id_feat is not None:
            out = self.last_linear(x - id_feat)
        else:
            out = self.last_linear(x)
        return out

    def forward(self, x):
        feats = self.features(x)
        logits = self.classifier(feats)
        return logits, feats


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        return checkpoint
    return checkpoint


def _strip_prefix(state_dict, prefix):
    if not prefix:
        return state_dict
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out


def load_xception_weights(model, weight_path, map_location="cpu"):
    """
    Load DeepfakeBench Xception weights.
    Handles common checkpoint formats and prefixes; uses strict=False.
    """
    checkpoint = torch.load(weight_path, map_location=map_location, weights_only=False)
    state_dict = _extract_state_dict(checkpoint)

    # Common prefixes from multi-GPU or wrapper modules
    state_dict = _strip_prefix(state_dict, "module.")
    state_dict = _strip_prefix(state_dict, "backbone.")
    state_dict = _strip_prefix(state_dict, "model.")
    state_dict = _strip_prefix(state_dict, "net.")

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    return missing_keys, unexpected_keys

