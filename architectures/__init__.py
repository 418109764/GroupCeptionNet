import torch

from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext18
from .googlenet import googlenet
from .shallow import shallow
from .vgg import vgg11, vgg13, vgg16, vgg19
from .alexnet import alexnet
from .mobilenetv2 import mobilenet_v2
from .inception import inception_v3
from .vit import vision_transformer
from .swin_transformer_v2 import swin_transformer_v2
from .mobilenetv4 import mobilenetv4_conv_small, mobilenetv4_conv_medium, mobilenetv4_conv_large
from .GroupCeptionNet import groupCeptionNet


MODELS = {
    "shallow": shallow,
    'resnet18': resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "resnext18": resnext18,
    "vgg11": vgg11,
    "vgg13": vgg13,
    "vgg16": vgg16,
    "vgg19": vgg19,
    "alexnet": alexnet,
    "googlenet": googlenet,
    "mobilenet_v2": mobilenet_v2,
    "inception_v3": inception_v3,
    "vision_transformer": vision_transformer,
    "swin_transformer_v2": swin_transformer_v2,
    "mobilenetv4_small": mobilenetv4_conv_small,
    "mobilenetv4_medium": mobilenetv4_conv_medium,
    "mobilenetv4_large": mobilenetv4_conv_large,
    "GroupCeptionNet": groupCeptionNet,
}


def cnn_model(model_name, pretrained=False, num_classes=(5, 5), weights_path=None):
    if model_name == "vision_transformer":
        model = MODELS[model_name](pretrained=pretrained,
                                   image_size=224,
                                   patch_size=16,
                                   num_classes=num_classes,
                                   dim=768,
                                   depth=12,
                                   heads=12,
                                   mlp_dim=3072,
                                   dropout=0.1,
                                   emb_dropout=0.1)
    elif model_name == "swin_transformer_v2":
        model = MODELS[model_name](pretrained=pretrained,
                                   img_size=224,
                                   num_classes=num_classes,)
    else:
        model = MODELS[model_name](pretrained=pretrained, num_classes=num_classes)

    if weights_path:
        try:
            model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
        except Exception:
            raise Exception("Error loading weights. You must train the model first.")

    if torch.cuda.is_available():
        model.cuda()

    return model
