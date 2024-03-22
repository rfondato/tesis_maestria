import cv2
import os
import re
import numpy as np
import time

def get_frames(file_name):
    f = file_name.split('.')[0]
    comp = f.split('_')
    return comp[len(comp)-2], comp[len(comp)-1]

def get_clean_filename(name):
    return re.sub(r'_[0-9]+_[0-9]+\..*$', '', name)

def remove_extension(file_name):
    return re.sub(r'\..*$', '', file_name)

def denormalize(keypoints):
    norm_factor = keypoints[-1]
    return keypoints[:len(keypoints)-1] * norm_factor

def add_contour_keypoints(frame, norm_keypoints) -> np.ndarray:
    keypoints = denormalize(norm_keypoints)
    keypoints, center = keypoints[:len(keypoints)-2], keypoints[-2:]
    angles = np.linspace(0, 2 * np.pi, len(keypoints), endpoint=False)
    cartesian_coords = np.zeros((len(keypoints), 2))
    cartesian_coords[:, 0] = keypoints * np.sin(angles) + center[0]
    cartesian_coords[:, 1] = keypoints * np.cos(angles) + center[1]
    cartesian_coords = cartesian_coords.astype(np.int32)
    for y,x in cartesian_coords:
        cv2.circle(frame, (x,y), 2, (0, 0, 255), 1)
    cv2.circle(frame, (int(center[1]), int(center[0])), 3, (255, 0, 0), 2)
    
    return cartesian_coords, center

def display_keypoints(preprocess_path:str, clip_path:str, file_name:str, augment_suffix:str = None):
    video_path = f"{clip_path}/{file_name}"
    cap = cv2.VideoCapture(video_path)

    keypoint_files = [f for f in os.listdir(preprocess_path) if (os.path.isfile(preprocess_path+f) and get_clean_filename(f) == remove_extension(file_name) + augment_suffix)]
    keypoint_data = [(get_frames(f)[0], np.load(preprocess_path+f)) for f in keypoint_files]
    keypoint_files_frames = [(f, get_frames(f)) for f in keypoint_files]
    start_frames = sorted([int(f[1][0]) for f in keypoint_files_frames])
    end_frames = sorted([int(f[1][1]) for f in keypoint_files_frames])

    frame_num = 0
    current_keypoint_data = None
    current_start_frame = -1
    current_end_frame = -1
    while cap.isOpened():
        success, frame = cap.read()
        
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))
        frame_num += 1
        
        if frame_num in start_frames:
            current_start_frame = frame_num
            current_end_frame = end_frames[start_frames.index(frame_num)]
            current_keypoint_data = list(filter(lambda kp: int(kp[0]) == current_start_frame, keypoint_data))[0][1]
        
        if current_keypoint_data is not None:
            if frame_num < current_end_frame:
                current_keypoints = current_keypoint_data[frame_num - current_start_frame]
                add_contour_keypoints(frame, current_keypoints)
                time.sleep(0.05)
            else:
                current_keypoint_data = None
                current_start_frame = -1
                current_end_frame = -1

        cv2.imshow("Keypoint detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

if __name__ == "__main__":
    name = 'person01_walking_d1_uncomp.avi'
    preprocess_path = './preprocess/walking/'
    clip_path = './input_data/train/walking/'
    augment_suffix = "_ht"

    display_keypoints(preprocess_path, clip_path, name, augment_suffix)
