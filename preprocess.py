import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
from tqdm import tqdm  # For progress tracking

# Set dataset path
dataset_path = "workoutfitness-video"
output_csv = "fitness_keypoints_full.csv"

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)

# Function to extract keypoints for all frames in a video
def extract_full_video_keypoints(video_path, label, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    keypoints_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit if video ends

        frame_count += 1
        if frame_count % frame_skip != 0:  # Process every 5th frame (change if needed)
            continue

        # Convert frame to RGB (MediaPipe requires RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform Pose Estimation
        results = pose.process(frame_rgb)

        # Extract keypoints if detected
        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.append(lm.x)  # X-coordinate (normalized)
                keypoints.append(lm.y)  # Y-coordinate (normalized)
                keypoints.append(lm.z)  # Depth coordinate

            keypoints.append(frame_count)  # Add frame number for sequence learning
            keypoints.append(label)  # Append exercise label
            keypoints_data.append(keypoints)

    cap.release()
    return keypoints_data

# Create an empty DataFrame to store keypoints
columns = [f"x_{i}" for i in range(33)] + [f"y_{i}" for i in range(33)] + [f"z_{i}" for i in range(33)] + ["frame_number", "label"]
pose_df = pd.DataFrame(columns=columns)

# Process all videos in dataset
for exercise in os.listdir(dataset_path):
    exercise_path = os.path.join(dataset_path, exercise)
    if os.path.isdir(exercise_path):  # Ensure it's a folder
        print(f"Processing exercise: {exercise}")

        for video_file in tqdm(os.listdir(exercise_path), desc=f"Processing {exercise}"):
            if video_file.endswith(".mp4"):  # Adjust based on dataset format
                video_path = os.path.join(exercise_path, video_file)
                keypoints = extract_full_video_keypoints(video_path, exercise)

                # Append results to DataFrame
                df_temp = pd.DataFrame(keypoints, columns=columns)
                pose_df = pd.concat([pose_df, df_temp], ignore_index=True)

# Save extracted keypoints to CSV
pose_df.to_csv(output_csv, index=False)
print(f"\n✅ Data processing complete! Keypoints saved to {output_csv}")