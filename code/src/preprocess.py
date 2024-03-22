import os
import cv2
import numpy as np
import torch as t
from ultralytics import YOLO
from tqdm import tqdm
from utils import timeit
import preprocess_cython
from multiprocessing import Pool

def do_backend_job (args):
    preprocessor, full_path, file, label = args
    if not os.path.isfile(full_path + file):
        return

    file_name = file.split('.')[0]
    preprocessor.preprocess_clip(file_name, full_path + file, label)

def cut_mask(mask: np.ndarray, bbox: np.ndarray, cut_x: int = None, cut_y: int = None, cut_x_to = True, cut_y_to = True):
    assert cut_x is not None or cut_y is not None, "Either cut_x or cut_y should be passed"
    
    new_mask = mask.copy()
    new_bbox = bbox.copy()

    cut_x = int(cut_x) if cut_x is not None else None
    cut_y = int(cut_y) if cut_y is not None else None

    if cut_x is not None:
        if cut_x_to:
            new_mask[:, cut_x:] = 0
            new_bbox[2] = cut_x
        else:
            new_mask[:, :cut_x] = 0
            new_bbox[0] = cut_x

    if cut_y is not None:
        if cut_y_to:
            new_mask[cut_y:, :] = 0
            new_bbox[3] = cut_y
        else:
            new_mask[:cut_y, :] = 0
            new_bbox[1] = cut_y

    return new_mask, new_bbox

def get_mass_center(binary_image):
    # calculate moments of binary image
    M = cv2.moments(binary_image)
    
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"]) if M["m00"] > 0 else 0
    cY = int(M["m01"] / M["m00"]) if M["m00"] > 0 else 0

    return np.array((cY, cX))

def get_contour_points_from_mask(mask, bbox, resolution=100):
    bbox_size = np.sqrt((bbox[0] - bbox[2])**2 + (bbox[1] - bbox[3])**2) if bbox is not None else 0
    center = get_mass_center(np.expand_dims(mask, 2)) #bbox[1] + (bbox[3] - bbox[1]) / 2, bbox[0] + (bbox[2] - bbox[0]) / 2

    values = preprocess_cython.get_contour_points_from_mask(mask, center[1], center[0], resolution, bbox_size)

    if bbox_size > 0:
        center = center / bbox_size

    return np.hstack((values, np.array(center), bbox_size))

def get_person_mask(yolo_results):
    if yolo_results is None:
        return None
    
    people_idx = t.argwhere(yolo_results.boxes.cls == 0)

    if len(people_idx) == 0:
        return None, None
    
    first_detected_person_idx = people_idx[0][0]
    mask = yolo_results.masks.data[first_detected_person_idx].cpu().numpy()
    bbox = yolo_results.boxes.xyxy[first_detected_person_idx].cpu().numpy()

    bbox[0] = bbox[0] * 640 / 160
    bbox[2] = bbox[2] * 640 / 160
    bbox[1] = bbox[1] * 480 / 120
    bbox[3] = bbox[3] * 480 / 120
    
    return mask, bbox

def get_person_contour_points(yolo_results, resolution = 100):
    mask, bbox = get_person_mask(yolo_results)

    if mask is None:
        return None
        
    return get_contour_points_from_mask(mask, bbox, resolution)

def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

class PreprocessedSequence:

    def __init__(self, max_size:int = None, output_folder:str = None, name:str = None) -> None:
        self.sequence = []
        self.max_size = max_size
        self.output_folder = output_folder
        self.name = name

    def reset(self) -> None:
        self.sequence = []

    def add(self, element: np.ndarray) -> None:
        if self.max_size is not None and len(self.sequence) >= self.max_size:
            self.sequence.pop(0)
        
        self.sequence.append(element)

    def get(self) -> np.ndarray:
        return np.array(self.sequence, np.float32)
    
    def save(self, frame_start:int, frame_end:int) -> None:

        create_folder_if_not_exists(self.output_folder)

        save_filename = f'{self.output_folder}/{self.name}_{frame_start}_{frame_end}.npy'

        with open(save_filename, 'wb') as f:
            np.save(f, self.get())
    
    def __len__(self) -> int:
        return len(self.sequence)
    
class CutAugmentStrategy:

    def __init__(self, cut_x:bool, cut_y:bool, cut_x_to:bool, cut_y_to:bool) -> None:
        self.cut_x = cut_x
        self.cut_y = cut_y
        self.cut_x_to = cut_x_to
        self.cut_y_to = cut_y_to

    def __call__(self, mask: np.ndarray, bbox: np.ndarray, center: np.ndarray) -> None:
        cut_x = center[1] if self.cut_x else None
        cut_y = center[0] if self.cut_y else None

        return cut_mask(mask, bbox, cut_x=cut_x, cut_y=cut_y, cut_x_to=self.cut_x_to, cut_y_to=self.cut_y_to)
        
class Preprocessor:

    def __init__(self, resolution:int = 100, seq_length:int = None, output_folder:str = None, augment_data:bool = False) -> None:
        self.model = YOLO('yolov8n-seg.pt')
        self.resolution = resolution
        self.output_folder = output_folder
        self.max_length = seq_length
        self.current_seq = None
        self.augment_data = augment_data
        if self.augment_data:
            self.augmented_seqs = [
                {"name": "ht", "sequence": None, "strategy": CutAugmentStrategy(cut_x=False, cut_y=True, cut_x_to=False, cut_y_to=True)},
                {"name": "hl", "sequence": None, "strategy": CutAugmentStrategy(cut_x=True, cut_y=False, cut_x_to=True, cut_y_to=False)},
                {"name": "hr", "sequence": None, "strategy": CutAugmentStrategy(cut_x=True, cut_y=False, cut_x_to=False, cut_y_to=False)}
            ]

    def process_realtime(self, frame):
        if self.current_seq is None:
            self.current_seq = PreprocessedSequence(max_size=self.max_length)

        results = self.model(frame, verbose=False)[0]
        features = get_person_contour_points(results, self.resolution)
        
        if features is None:
            self.current_seq.reset()
            return
        
        self.current_seq.add(features)
        return self.current_seq.get()
    
    def preprocess_folder(self, folder, label, parallel_jobs = 1):
        full_path = f'{folder}/{label}/'
        
        def do_job (f):
            if not os.path.isfile(full_path + f):
                return

            file_name = f.split('.')[0]
            self.preprocess_clip(file_name, full_path + f, label)

        if parallel_jobs <= 1:
            for f in tqdm(os.listdir(full_path), desc=f"Processing clips for {label}"):
                do_job(f)
        else:
            pool = Pool(processes=parallel_jobs)
            files = [(self, full_path, f, label) for f in os.listdir(full_path)]

            for _ in tqdm(pool.imap_unordered(do_backend_job, files), total=len(files)):
                pass
    
    def preprocess_clip(self, clip_name, clip_path, label):
        cap = cv2.VideoCapture(clip_path)
        frame_start = None
        frame_num = 0
        self.current_seq = PreprocessedSequence(output_folder=(self.output_folder + label), name=clip_name)
        self._create_augmented_sequences(label, clip_name)
        
        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            frame_num += 1

            results = self.model(frame, verbose=False)[0]
            features = get_person_contour_points(results, self.resolution)

            if features is None:
                if len(self.current_seq) > 0:         
                    self.current_seq.save(frame_start, frame_num)
                    self._save_augmented_sequences(frame_start, frame_num)
                    frame_start = None
                    self.current_seq.reset()
                    self._reset_augmented_sequences()
                
                continue

            if frame_start is None:
                frame_start = frame_num

            self.current_seq.add(features)
            if self.augment_data:
                mask, bbox = get_person_mask(results)
                self._augment_sequences(mask, bbox)
            
        if len(self.current_seq) > 0:
            self.current_seq.save(frame_start, frame_num)
            self._save_augmented_sequences(frame_start, frame_num)

    def _create_augmented_sequences(self, label:str, name:str):
        if not self.augment_data:
            return
        
        for s in self.augmented_seqs:
            s["sequence"] = PreprocessedSequence(output_folder=(self.output_folder + label), name=f"{name}_{s['name']}")

    def _augment_sequences(self, mask: np.ndarray, bbox: np.ndarray):
        if not self.augment_data:
            return
            
        center = get_mass_center(np.expand_dims(mask, 2))

        for s in self.augmented_seqs:
            new_mask, new_bbox = s["strategy"](mask, bbox, center)
                
            features = get_contour_points_from_mask(new_mask, new_bbox, self.resolution)
            assert features is not None, "Augmented features cannot be none if original features were not None"
            s["sequence"].add(features)
    
    def _save_augmented_sequences(self, frame_start: int, frame_end: int):
        if not self.augment_data:
            return

        for s in self.augmented_seqs:
            s["sequence"].save(frame_start, frame_end)

    def _reset_augmented_sequences(self):
        if not self.augment_data:
            return
        
        for s in self.augmented_seqs:
            s["sequence"].reset()

if __name__ == '__main__':
    input_folder = "./input_data/train/"
    output_folder = "./preprocess/"
    #labels = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
    labels = ['walking']
    resolution = 100

    preprocessor = Preprocessor(resolution, output_folder=output_folder, augment_data=True)

    for label in tqdm(labels, desc="Processing labels"):
        preprocessor.preprocess_folder(input_folder, label, parallel_jobs=4)
