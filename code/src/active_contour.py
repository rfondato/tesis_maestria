
import numpy as np

N4_ENCODING = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])

class ActiveContour:

    def __init__(self, bins: list[int], feature_ranges: list[float], max_iter = 100) -> None:
        self.binned_image = None
        self.level_set = None
        self.speed = None
        self.inner_histogram = np.zeros((bins[0], bins[1]), dtype=np.int32)
        self.inner_count = 0
        self.outer_histogram = np.zeros((bins[0], bins[1]), dtype=np.int32)
        self.outer_count = 0
        self.features_bin_size = [feature_ranges[i] / bins[i] for i in range(len(bins))]
        self.initialized = False
        self.max_iter = max_iter

    def apply (self, image: np.ndarray, mask: np.ndarray):
        self._create_binned_image(image)

        if not self.initialized:
            self._initialize(mask)
        
        for _ in range(self.max_iter):
            self._calculated_speed()
            self._switch_in_edge_points()
            self._switch_l_in_to_internal()
            self._switch_out_edge_points()
            self._switch_l_out_to_external()

            if self._should_end():
                break
        
        resultingMask = np.zeros((image.shape[0], image.shape[1]))
        resultingMask[self.level_set == 1] = 1

        return resultingMask

    def _switch_in_edge_points(self):
        coords = np.argwhere((self.level_set == 1) & self.speed)

        for c in coords:
            self._switch_in(c)

    def _switch_l_in_to_internal(self):
        coords = np.argwhere(self.level_set == -1)

        for c in coords:
            n4 = self._get_n4(c[0], c[1], self.level_set)
            if np.all(n4 < 0):
                self.level_set[c[0], c[1]] = -3
                y,v = self.binned_image[c[0], c[1],0], self.binned_image[c[0], c[1],1]
                #self.inner_histogram[y, v] += 1
                #self.inner_count += 1

    def _switch_out_edge_points(self):
        coords = np.argwhere((self.level_set == -1) & (~self.speed))

        for c in coords:
            self._switch_out(c)

    def _switch_l_out_to_external(self):
        coords = np.argwhere(self.level_set == 1)

        for c in coords:
            n4 = self._get_n4(c[0], c[1], self.level_set)
            if np.all(n4 > 0):
                self.level_set[c[0], c[1]] = 3
                y,v = self.binned_image[c[0], c[1],0], self.binned_image[c[0], c[1],1]
                #self.outer_histogram[y, v] += 1
                #self.outer_count += 1

    def _switch_in(self, x):
        self.level_set[x[0], x[1]] = -1
        n4 = self._get_n4(x[0], x[1], self.level_set)
        to_lout = x + N4_ENCODING[n4 == 3, :]

        for c in to_lout:
            self.level_set[c[0], c[1]] = 1
            y,v = self.binned_image[c[0], c[1],0], self.binned_image[c[0], c[1],1]
            #self.outer_histogram[y, v] -= 1
            #self.outer_count -= 1
    
    def _switch_out(self, x):
        self.level_set[x[0], x[1]] = 1
        n4 = self._get_n4(x[0], x[1], self.level_set)
        to_lin = x + N4_ENCODING[n4 == -3, :]
        
        for c in to_lin:
            self.level_set[c[0], c[1]] = -1
            y,v = self.binned_image[c[0], c[1],0], self.binned_image[c[0], c[1],1]
            #self.inner_histogram[y, v] -= 1
            #self.inner_count -= 1

    def _should_end(self):
        return np.all(self.speed[self.level_set < 0]) and not np.any(self.speed[self.level_set > 0])

    def _create_binned_image(self, image: np.ndarray):
        self.binned_image = np.zeros((image.shape[0], image.shape[1], 2), dtype=np.int8)
        self.binned_image[:, :, 0] = (image[:, :, 0] // self.features_bin_size[0]).astype(np.int8)
        self.binned_image[:, :, 1] = (image[:, :, 2] // self.features_bin_size[1]).astype(np.int8)

    def _initialize(self, mask: np.ndarray):
        self.level_set = np.zeros((self.binned_image.shape[0], self.binned_image.shape[1]), dtype=np.int8)
        self.speed = np.zeros((self.binned_image.shape[0], self.binned_image.shape[1]), dtype=np.bool_)
        
        for i in range(self.binned_image.shape[0]):
            for j in range(self.binned_image.shape[1]):
                y,v = self.binned_image[i,j,0], self.binned_image[i,j,1]
                value = mask[i,j]
                n4 = self._get_n4(i, j, mask)
                if self._is_internal(value, n4):
                    self.level_set[i, j] = -3
                    self.inner_histogram[y, v] += 1
                    self.inner_count += 1
                elif self._is_l_in(value, n4):
                    self.level_set[i, j] = -1
                elif self._is_l_out(value, n4):
                    self.level_set[i, j] = 1
                else: # It's external
                    self.level_set[i, j] = 3
                    self.outer_histogram[y, v] += 1
                    self.outer_count += 1
        
        self.initialized = True

    def _calculated_speed(self):
        edge_coords = np.argwhere((self.level_set == 1) | (self.level_set == -1))

        for c in edge_coords:
            y,v = self.binned_image[c[0],c[1],0], self.binned_image[c[0], c[1],1]
            prob_in = self.inner_histogram[y, v] / self.inner_count
            prob_out = self.outer_histogram[y, v] / self.outer_count
            self.speed[c[0], c[1]] = prob_in > prob_out

    def _get_n4(self, i: int, j:int, array: np.ndarray):
        n4 = np.zeros(4)

        for c in range(4):
            i2 = i + N4_ENCODING[c, 0]
            j2 = j + N4_ENCODING[c, 1]
            n4[c] = array[i2, j2] if self._is_within_coords(i2, j2, array.shape[0], array.shape[1]) else 0
        
        return n4

    def _is_within_coords(self, i, j, max_i, max_j):
        return i >= 0 and i < max_i and j >= 0 and j < max_j

    def _is_internal(self, value, n4: np.ndarray):
        return value == 1 and (n4 == 1).all()
    
    def _is_l_in(self, value, n4: np.ndarray):
        return value == 1 and (n4 == 0).any()
    
    def _is_l_out(self, value, n4: np.ndarray):
        return value == 0 and (n4 == 1).any()
        
if __name__ == "__main__":
    import cv2
    from ultralytics import YOLO
    import torch as t

    cap = cv2.VideoCapture('./input_data/train/jogging/person01_jogging_d1_uncomp.avi')
    model = YOLO('yolov8n-seg.pt')
    active_contour = ActiveContour(bins=[32, 32], feature_ranges=[255, 255], max_iter=10)

    def get_person_mask(yolo_results):
        if yolo_results is None:
            return None
        
        people_idx = t.argwhere(yolo_results.boxes.cls == 0)

        if len(people_idx) == 0:
            return None
        
        return np.moveaxis(yolo_results.masks.data[people_idx[0]].cpu().numpy(), 0, 2)

    frame_num = -1
    mask = None

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        frame_num += 1

        if frame_num % 10 == 0:
            results = model(frame, verbose=False)[0]
            mask = get_person_mask(results)
            
            if mask is not None:
                mask = mask.squeeze()

            active_contour = ActiveContour(bins=[32, 32], feature_ranges=[255, 255], max_iter=10)

        if mask is None:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        frame = cv2.resize(frame, (640, 480))
        mask = active_contour.apply(frame, mask)
                
        cv2.imshow("Inference", mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
