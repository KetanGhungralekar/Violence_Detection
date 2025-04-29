import pandas as pd
import cv2
import os
from tqdm import tqdm

import random
import shutil
from ultralytics import YOLO
import glob

# Load labels from CSV files
violent_df = pd.read_csv(r"C:\Users\mitta\OneDrive - iiit-b\Documents\PE_Sushree_Ma'am\A-Dataset-for-Automatic-Violence-Detection-in-Videos\violence-detection-dataset\violent-action-classes.csv", delimiter=';', encoding='utf-8')
nonviolent_df = pd.read_csv(r"C:\Users\mitta\OneDrive - iiit-b\Documents\PE_Sushree_Ma'am\A-Dataset-for-Automatic-Violence-Detection-in-Videos\violence-detection-dataset\nonviolent-action-classes.csv", delimiter=';', encoding='utf-8')
print(violent_df)

# Assign labels
video_labels = {}

# Assuming CSVs have a column named 'FILE' for video names
for video in violent_df["FILE"]:
    video_labels[video] = 1  # 1 for violent

for video in nonviolent_df["FILE"]:
    video_labels[video] = 0  # 0 for non-violent

video_root = r"C:\Users\mitta\OneDrive - iiit-b\Documents\PE_Sushree_Ma'am\A-Dataset-for-Automatic-Violence-Detection-in-Videos\violence-detection-dataset"
output_root = r"C:\Users\mitta\OneDrive - iiit-b\Documents\PE_Sushree_Ma'am\A-Dataset-for-Automatic-Violence-Detection-in-Videos"

# Categories
categories = ["violent", "non-violent"]

# Frame extraction settings
FRAME_RATE = 5  # Extract every 5th frame

def extract_frames(video_path, output_folder, frame_rate=5):
    """Extracts frames from a video and saves them in the output folder."""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_rate == 0:
            frame_filename = os.path.join(output_folder, f"{os.path.basename(video_path)}_frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()

# Process each category
for category in categories:
    category_path = os.path.join(video_root, category)

    for cam_folder in os.listdir(category_path):
        cam_path = os.path.join(category_path, cam_folder)

        if not os.path.isdir(cam_path):
            continue  # Skip if not a folder

        # Create corresponding output folder
        output_folder = os.path.join(output_root, category)
        os.makedirs(output_folder, exist_ok=True)

        # Process videos in the cam folder
        for video_file in tqdm(os.listdir(cam_path), desc=f"Processing {category}/{cam_folder}"):
            video_path = os.path.join(cam_path, video_file)

            if video_file.endswith(".mp4"):  # Ensure it's a video
                extract_frames(video_path, output_folder, FRAME_RATE)

# Define paths
dataset_root = video_root
split_ratio = 0.8  # 80% train, 20% validation

# Create folders
for split in ["train", "val"]:
    for category in ["violent", "non-violent"]:
        os.makedirs(os.path.join(dataset_root, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(dataset_root, split, "labels"), exist_ok=True)

# Function to split images
def split_data(category):
    frames_path = os.path.join(output_root, category)
    images = [f for f in os.listdir(frames_path) if f.endswith(".jpg")]
    random.shuffle(images)

    train_count = int(len(images) * split_ratio)
    train_images = images[:train_count]
    val_images = images[train_count:]

    for img in train_images:
        shutil.move(os.path.join(frames_path, img), os.path.join(dataset_root, "train/images", img))
    
    for img in val_images:
        shutil.move(os.path.join(frames_path, img), os.path.join(dataset_root, "val/images", img))

# Process both categories
split_data("violent")
split_data("non-violent")

print("Dataset split completed!")

violent_df["label"] = 1
nonviolent_df["label"] = 0
df = pd.concat([violent_df, nonviolent_df])

df.rename(columns={"FILE": "Video"}, inplace=True)

# Generate YOLO labels
for index, row in df.iterrows():
    video_name = row["Video"]
    label = row["label"]
    
    for split in ["train", "val"]:
        images_folder = os.path.join(dataset_root, split, "images")
        labels_folder = os.path.join(dataset_root, split, "labels")

        for image_file in os.listdir(images_folder):
            if video_name in image_file:
                label_file = os.path.join(labels_folder, image_file.replace(".jpg", ".txt"))
                with open(label_file, "w") as f:
                    f.write(f"{label} 0.5 0.5 1.0 1.0\n")

# Load YOLOv8 model
# model = YOLO("yolov8n.pt")

# Train the model
# model.train(data=r"C:\Users\mitta\OneDrive - iiit-b\Documents\PE_Sushree_Ma'am\A-Dataset-for-Automatic-Violence-Detection-in-Videos\data.yaml", epochs=25, imgsz=224)

# Load trained model
model = YOLO('runs/detect/train7/weights/best.pt')

# Validate model performance
metrics = model.val()
print(f"mAP50: {metrics.box.map:.4f}")  # mAP at 50% IoU
