import os
import cv2
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split


def rotate_and_fill(pil_image, angle):
    """
    Rotate the PIL image and fill the border with white color to avoid black edges
    :param pil_image: Input PIL image
    :param angle: Rotation angle (degrees)
    :return: Rotated PIL image with white background filling
    """
    image_np = np.array(pil_image)

    # Get the center coordinates of the image
    center = tuple(np.array(image_np.shape[1::-1]) / 2)

    # Rotate the image
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image_np, rotation_matrix, image_np.shape[1::-1], flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    # Convert numpy array back to PIL.Image
    return Image.fromarray(rotated)


# Define image augmentation pipeline
transform = transforms.Compose([
    transforms.Resize((140, 140)),
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: rotate_and_fill(img, np.random.uniform(-30, 30))),  # Random rotation (-30° to 30°)
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomVerticalFlip(),    # Random vertical flip
    transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 2.0), saturation=(0.5, 2.0)),  # Color jitter
])

# Create augmented image directory if it does not exist
if not os.path.exists('dataset/augImages'):
    os.mkdir('dataset/augImages')

# Read the original CSV file
df = pd.read_csv('dataset/dataset.csv')

# Split 20% of the data as test set (stratified split to keep class distribution)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=122, stratify=df['predominant_stress'])
print(f"Number of test images: {len(test_df)}")

# Calculate the number of images per class in training set
class_counts = train_df['predominant_stress'].value_counts()

# Calculate the average number of images across all classes
average_count = class_counts.mean()

# Calculate augmentation multiplier (ratio to average count * 5, rounded to integer)
augment_multiplier = {category: round(5 * average_count / count) for category, count in class_counts.items()}

rows_list = []  # List to collect new augmented training rows
test_rows_list = []  # List to collect test rows

# Augment training set images
for index, row in train_df.iterrows():
    image_path = os.path.join('dataset/images', f"{row['id']}.jpg")
    image = Image.open(image_path)

    # Get augmentation times based on pre-calculated multiplier
    augment_times = augment_multiplier[row['predominant_stress']]

    for i in range(augment_times):
        augmented_image = transform(image)
        # Generate new unique ID for augmented image
        new_id = max(df['id']) + len(rows_list) + 1

        # Save augmented image
        augmented_image_path = os.path.join('dataset/augImages', f"{new_id}.jpg")
        augmented_image.save(augmented_image_path)

        # Add new row to the list
        new_row = row.copy()
        new_row['id'] = new_id
        rows_list.append(new_row)

# Copy test set images to augImages directory and collect test rows
for _, row in test_df.iterrows():
    # Copy test images to the augmentation directory
    src_path = os.path.join('dataset/images', f"{row['id']}.jpg")
    dest_path = os.path.join('dataset/augImages', f"{row['id']}.jpg")
    if not os.path.exists(dest_path):
        shutil.copy(src_path, dest_path)
    test_rows_list.append(row)

# Create new DataFrames from the collected rows
aug_df = pd.DataFrame(rows_list)
aug_test_df = pd.DataFrame(test_rows_list)

# Save augmented training set and test set to CSV files
aug_df.to_csv('dataset/augDataset.csv', index=False)
aug_test_df.to_csv('dataset/augTestDataset.csv', index=False)