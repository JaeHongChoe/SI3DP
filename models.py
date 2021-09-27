import torch
import torch.nn as nn
import geffnet
from resnest.torch import resnest101
from pretrainedmodels import se_resnext101_32x4d

'''
class Network_name(nn.Module):
    # See the network creation section below.
    def __init__(self, net_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128], pretrained=False):
        # initialization
        enet_type : Network name from argument
        out_dim   : output layer size
        n_meta_dim      : mlp size (2 basic layers)
        pretrained      : Will you use a pre-trained model?

    def extract(self, x):
        # Extract the results of the base network (image deep features)
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        # Get final network results (Include fc_layer)
'''

class Effnet_MMC(nn.Module):
    def __init__(self, enet_type, out_dim, n_meta_dim=[512, 128], pretrained=False):
        super(Effnet_MMC, self).__init__()
        # efficient net Model
        self.enet = geffnet.create_model(enet_type, pretrained=pretrained)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        in_ch = self.enet.classifier.in_features

        self.myfc = nn.Sequential(
            nn.Linear(in_ch, n_meta_dim[0]),
            nn.BatchNorm1d(n_meta_dim[0]),
            # swish activation function
            Swish_Module(),
            nn.Dropout(p=0.3),
            nn.Linear(n_meta_dim[0], n_meta_dim[1]),
            nn.BatchNorm1d(n_meta_dim[1]),
            Swish_Module(),
            nn.Linear(n_meta_dim[1], out_dim)
        )


        self.enet.classifier = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)

        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))

        out /= len(self.dropouts)
        return out

class Effnet_MMC_Multitask(nn.Module):
    def __init__(self, enet_type, out_dim, out_dim2, pretrained=False):
        super(Effnet_MMC_Multitask, self).__init__()
        # efficient net 모델
        self.enet = geffnet.create_model(enet_type, pretrained=pretrained)

        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        in_ch = self.enet.classifier.in_features

        self.myfc_1 = nn.Linear(in_ch, out_dim)
        self.myfc_2 = nn.Linear(in_ch, out_dim2)

        self.enet.classifier = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)

        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out1 = self.myfc_1(dropout(x))
                out2 = self.myfc_2(dropout(x))

            else:
                out1 += self.myfc_1(dropout(x))
                out2 += self.myfc_2(dropout(x))

        out1 /= len(self.dropouts)
        out2 /= len(self.dropouts)

        return out1, out2

class Effnet_MMC_Multi_Modal(nn.Module):
    def __init__(self, enet_type, out_dim, out_dim2, pretrained=False):
        super(Effnet_MMC_Multi_Modal, self).__init__()

        # efficient net 모델
        self.enet = geffnet.create_model(enet_type, pretrained=pretrained)
        if out_dim2 == 5:
            self.enet2 = geffnet.create_model(enet_type, pretrained=pretrained)
            self.enet2.classifier = nn.Identity()
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        in_ch = self.enet.classifier.in_features

        self.fc_device_closefull = nn.Linear(in_ch*2, out_dim)
        self.fc_quality_closefull = nn.Linear(in_ch*2, out_dim2)
        self.fc_quality_close = nn.Linear(in_ch, out_dim2)

        self.enet.classifier = nn.Identity()


    def extract(self, x, x2):
        x = self.enet(x)
        x2 = self.enet(x2)
        return x, x2

    def forward(self, close, full):
        close = self.enet(close).squeeze(-1).squeeze(-1)
        full = self.enet(full).squeeze(-1).squeeze(-1)
        close_full = torch.cat((close, full), dim=1)

        for i, dropout in enumerate(self.dropouts):
            out1 = self.fc_device_closefull(dropout(close_full))
            out2 = self.fc_quality_closefull(dropout(close_full))
            out3 = self.fc_quality_close(dropout(close))

        out1 /= len(self.dropouts)
        out2 /= len(self.dropouts)
        out3 /= len(self.dropouts)

        return out1, out2, out3

class Effnet_MMC_Multi_Modal_Single_Task(nn.Module):
    def __init__(self, enet_type, out_dim, n_meta_dim=[512, 128], pretrained=False):
        super(Effnet_MMC_Multi_Modal_Single_Task, self).__init__()

        # efficient net 모델
        self.enet = geffnet.create_model(enet_type, pretrained=pretrained)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        in_ch = self.enet.classifier.in_features

        self.fc_device_closefull = nn.Linear(in_ch*2, out_dim)
        self.fc_device_close = nn.Linear(in_ch, out_dim)

        self.enet.classifier = nn.Identity()

    def extract(self, x, x2):
        x = self.enet(x)
        x2 = self.enet(x2)
        return x, x2

    def forward(self, close, full):
        close = self.enet(close).squeeze(-1).squeeze(-1)
        full = self.enet(full).squeeze(-1).squeeze(-1)
        close_full = torch.cat((close, full), dim=1)

        for i, dropout in enumerate(self.dropouts):

            out = self.fc_device_closefull(dropout(close_full))
            out2 = self.fc_device_close(dropout(close))
            # if i == 0:
            #     out = self.fc_device_closefull(dropout(close_full))
            # else:
            #     out += self.fc_device_closefull(dropout(close_full))

        out /= len(self.dropouts)
        out2 /= len(self.dropouts)

        return out, out2

class Resnest_MMC(nn.Module):
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128], pretrained=False):
        super(Resnest_MMC, self).__init__()
        self.n_meta_features = n_meta_features
        self.enet = resnest101(pretrained=pretrained)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        in_ch = self.enet.fc.in_features
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                Swish_Module(),
                nn.Dropout(p=0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                Swish_Module(),
            )
            in_ch += n_meta_dim[1]
        self.myfc = nn.Sequential(
            nn.Linear(in_ch, out_dim),
            nn.BatchNorm1d(n_meta_dim[0]),
            # swish activation function
            Swish_Module(),
            nn.Dropout(p=0.3),
            nn.Linear(n_meta_dim[0], n_meta_dim[1]),
            nn.BatchNorm1d(n_meta_dim[1]),
            Swish_Module(),
        )
        self.enet.fc = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        return out


class Seresnext_MMC(nn.Module):
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128], pretrained=False):
        super(Seresnext_MMC, self).__init__()
        self.n_meta_features = n_meta_features
        if pretrained:
            self.enet = se_resnext101_32x4d(num_classes=1000, pretrained='imagenet')
        else:
            self.enet = se_resnext101_32x4d(num_classes=1000, pretrained=None)
        self.enet.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        in_ch = self.enet.last_linear.in_features
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                Swish_Module(),
                nn.Dropout(p=0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                Swish_Module(),
            )
            in_ch += n_meta_dim[1]
        self.myfc = nn.Sequential(
            nn.Linear(in_ch, out_dim),
            nn.BatchNorm1d(n_meta_dim[0]),
            # swish activation function
            Swish_Module(),
            nn.Dropout(p=0.3),
            nn.Linear(n_meta_dim[0], n_meta_dim[1]),
            nn.BatchNorm1d(n_meta_dim[1]),
            Swish_Module(),
        )
        self.enet.last_linear = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        return out

sigmoid = nn.Sigmoid()

# swish activation function
# sigmoid에 x를 곱한 형태
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)

