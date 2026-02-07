import torch
from contourpy.util.data import simple
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
import json
import numpy as np
import cv2

from architectures.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext18
from architectures.googlenet import googlenet
from architectures.vgg import vgg11, vgg13, vgg16, vgg19
from architectures.alexnet import alexnet
from architectures.mobilenetv2 import mobilenet_v2
from architectures.inception import inception_v3
from architectures.vit import vision_transformer
from architectures.swin_transformer_v2 import swin_transformer_v2
from architectures.mobilenetv4 import mobilenetv4_conv_small, mobilenetv4_conv_medium, mobilenetv4_conv_large
from architectures.AirNeXt import airnext


# Optional model list
MODEL_WEIGHTS = {
    'resnet18': {
        'model': resnet18,
        'weights': 'results/B32_SP622_3_0/3_1_resnet18_b32_sp622o_sgd1e-4/net_weights.pth'
    },
    'resnet34': {
        'model': resnet34,
        'weights': 'path_to_resnet34_weights.pth'
    },
    'resnet50': {
        'model': resnet50,
        'weights': 'path_to_resnet50_weights.pth'
    },
    'resnet101': {
        'model': resnet101,
        'weights': 'path_to_resnet101_weights.pth'
    },
    'resnet152': {
        'model': resnet152,
        'weights': 'path_to_resnet152_weights.pth'
    },
    'resnext18': {
        'model': resnext18,
        'weights': 'path_to_resnext18_weights.pth'
    },
    'vgg11': {
        'model': vgg11,
        'weights': 'path_to_vgg11_weights.pth'
    },
    'vgg13': {
        'model': vgg13,
        'weights': 'results/B32_SP622_3_0/3_2_vgg13_b32_sp622o_sgd1e-4/net_weights.pth'
    },
    'vgg16': {
        'model': vgg16,
        'weights': 'path_to_vgg16_weights.pth'
    },
    'vgg19': {
        'model': vgg19,
        'weights': 'path_to_vgg19_weights.pth'
    },
    'alexnet': {
        'model': alexnet,
        'weights': 'path_to_alexnet_weights.pth'
    },
    'googlenet': {
        'model': googlenet,
        'weights': 'path_to_googlenet_weights.pth'
    },
    'mobilenet_v2': {
        'model': mobilenet_v2,
        'weights': 'path_to_mobilenet_v2_weights.pth'
    },
    'inception_v3': {
        'model': inception_v3,
        'weights': 'results/B32_SP622_3_0/3_1_inception_v3_1_b32_sp622o_sgd1e-4/net_weights.pth'
    },
    'vision_transformer': {
        'model': vision_transformer,
        'weights': 'path_to_vision_transformer_weights.pth'
    },
    'swin_transformer_v2': {
        'model': swin_transformer_v2,
        'weights': 'path_to_swin_transformer_v2_weights.pth'
    },
    'mobilenetv4_small': {
        'model': mobilenetv4_conv_small,
        'weights': 'path_to_mobilenetv4_small_weights.pth'
    },
    'mobilenetv4_medium': {
        'model': mobilenetv4_conv_medium,
        'weights': 'path_to_mobilenetv4_medium_weights.pth'
    },
    'mobilenetv4_large': {
        'model': mobilenetv4_conv_large,
        'weights': 'path_to_mobilenetv4_large_weights.pth'
    },
    'airnext': {
        'model': airnext,
        'weights': 'results/AirNet/AirNet_GC/1GC_airnet_b32_sp622o_Adam1e-4/net_weights.pth'
    },
}

# Example: Get model and weight path by model name
def get_model_and_weights(model_name):
    if model_name in MODEL_WEIGHTS:
        return MODEL_WEIGHTS[model_name]['model'], MODEL_WEIGHTS[model_name]['weights']
    else:
        raise ValueError(f"Model '{model_name}' not found in registered models.")


def preprocess_image(image_path, input_size=(224, 224)):
    """Preprocess input image"""
    transform = Compose([
        Resize(input_size),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0), image


def load_model(model_constructor, weights_path, num_classes=2, device="cuda"):
    """Load model and weights"""
    # Define model structure
    model = model_constructor(num_classes=num_classes).to(device)

    # Load weights
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def load_labelme_json(json_path):
    """
    Parse LabelMe JSON file and extract polygon points
    :param json_path: Path to JSON file
    :return: List of polygon points
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    polygons = []
    for shape in data["shapes"]:
        if shape["shape_type"] == "polygon":
            # Extract polygon points
            polygons.append(np.array(shape["points"], dtype=np.int32))
    return polygons


def calculate_iou(gt_mask, heatmap_binary):
    """
    Calculate IoU (Intersection over Union)
    :param gt_mask: Ground truth binary mask
    :param heatmap_binary: Binarized heatmap
    :return: IoU value
    """
    # Calculate intersection and union
    intersection = np.logical_and(gt_mask > 0, heatmap_binary > 0)
    union = np.logical_or(gt_mask > 0, heatmap_binary > 0)
    iou = np.sum(intersection) / np.sum(union)

    return iou


def generate_heatmap_with_iou(image_path, model_constructor, weights_path, json_path, target_layer_name="layer4.1", threshold=0.7):
    """
    Generate Grad-CAM heatmap with annotated polygon overlay and calculate IoU
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = load_model(model_constructor, weights_path, device=device)

    # Initialize Grad-CAM
    cam_extractor = GradCAM(model, target_layer=target_layer_name)

    # Load image
    input_tensor, original_image = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)

    # Load annotated polygons and generate ground truth mask
    polygons = load_labelme_json(json_path)
    gt_mask = np.zeros(original_image.size[::-1], dtype=np.uint8)  # (height, width)
    for polygon in polygons:
        cv2.fillPoly(gt_mask, [polygon], color=255)

    # Generate prediction
    output = model(input_tensor)
    class_idx = output.squeeze(0).argmax().item()  # Get predicted class index

    # Extract Grad-CAM activation map
    activation_map = cam_extractor(class_idx, output)
    activation_map = activation_map[0].squeeze().cpu().numpy()  # Ensure 2D array

    # Resize heatmap to original image size
    heatmap_resized = cv2.resize(
        activation_map,
        (original_image.size[0], original_image.size[1]),  # (width, height)
        interpolation=cv2.INTER_CUBIC
    )

    # Normalize heatmap
    heatmap_normalized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())
    heatmap_binary = (heatmap_normalized >= threshold).astype(np.uint8)  # Binarize with threshold

    # Calculate IoU
    iou = calculate_iou(gt_mask > 0, heatmap_binary)

    # Convert heatmap to PIL Image
    from torchvision.transforms.functional import to_pil_image
    heatmap_image = to_pil_image(heatmap_resized, mode="F")

    # Overlay heatmap on original image
    result = overlay_mask(original_image, heatmap_image, alpha=0.5)

    # Draw annotated polygons on heatmap
    draw = ImageDraw.Draw(result)
    for polygon in polygons:
        draw.polygon([tuple(point) for point in polygon], outline="red", width=6)

    # Draw contours of thresholded region in heatmap
    contours, _ = cv2.findContours(heatmap_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        points = [tuple(point[0]) for point in contour]
        draw.line(points + [points[0]], fill="blue", width=6)  # Draw contour with blue color

    # Display IoU value
    # Define font path and font size
    font_size = 32  # Set font size
    # Load font
    font = ImageFont.truetype("arialbd.ttf", font_size)  # Replace with font path on your system
    # Bold font
    draw.text((10, 10), f"IoU: {iou:.2f}", fill="white", font=font, stroke_width=2, stroke_fill="black")

    # Display result
    plt.imshow(result)
    plt.axis('off')
    plt.show()

    # Save result
    model_name = model_constructor.__name__
    result.save(f"{model_name}_IMG_2936_gradcam_iou_output.png")
    print(f"Heatmap with IoU saved as {model_name}_gradcam_iou_output.png")


if __name__ == "__main__":
    selected_model = 'inception_v3'  # Select model name
    model_constructor, weights_path = get_model_and_weights(selected_model)
    image_path = "process_dataset/IMG_2936.jpg"  # Input image path (germinated:IMG_2936, non-germinated: IMG_2937, IMG_4364, IMG_4377)
    json_path = "results/heatmap_and_iou/IMG_2936.json"  # Path to LabelMe annotation JSON file
    target_layer_name = "Mixed_7c"  # Replace with your target layer name (features, layer4.1, Mixed_7c)

    generate_heatmap_with_iou(image_path, model_constructor, weights_path, json_path, target_layer_name=target_layer_name)