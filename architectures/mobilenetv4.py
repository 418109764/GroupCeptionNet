# mobilenetv4.py
import timm

def mobilenetv4_conv_small(pretrained=False, num_classes=2):
    model = timm.create_model(
        "hf_hub:timm/mobilenetv4_conv_small.e2400_r224_in1k",
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model

def mobilenetv4_conv_medium(pretrained=False, num_classes=2):
    model = timm.create_model(
        'hf_hub:timm/mobilenetv4_conv_medium.e500_r224_in1k',
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model

def mobilenetv4_conv_large(pretrained=False, num_classes=2):
    model = timm.create_model(
        'hf_hub:timm/mobilenetv4_conv_large.e500_r256_in1k',
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model
