#%% 
import cv2
import numpy as np

def get_mass_center(binary_image):
    # calculate moments of binary image
    M = cv2.moments(binary_image)
    
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return cY, cX

def get_contour_points(mask, resolution=100):
    center = get_mass_center(mask)
    coords = np.argwhere(mask.squeeze() == 1) - np.array(center)
    polar_coords = np.array((np.linalg.norm(coords, axis=1), np.arctan2(coords[:, 0], coords[:, 1]))).T
    polar_coords = polar_coords[~np.isnan(polar_coords).any(axis=1), :]
    chosen_angles = np.linspace(-np.pi, np.pi, resolution, endpoint=False)
    angles = np.expand_dims(chosen_angles, 1)
    angles = angles.repeat(polar_coords.shape[0], axis=1)
    diffs = np.abs(polar_coords[:, 1] - angles)
    idx = np.argmin(diffs, axis=1)
    return chosen_angles, polar_coords[idx, 0]
    
dummy_mask = np.zeros((16, 16, 1), np.uint8)
#cv2.rectangle(dummy_mask, (2, 2), (14, 14), 1, 1)
#cv2.rectangle(dummy_mask, (5, 5), (11, 11), 1, 1)

dummy_mask[np.random.random(dummy_mask.shape) <= 0.8] = 1

points = get_contour_points(dummy_mask, 8)

#cv2.imshow("Mask", dummy_mask)
#cv2.waitKey(10000)

# %%
