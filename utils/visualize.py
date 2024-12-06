
import numpy as np
import cv2 as cv


def visualize_depth(depth: np.ndarray):
    """
    Visualize depth map.
    
    Args:
        depth: Depth map in [H, W].
        
    Returns:
        Image with depth map.
    """
    depth = depth.astype(np.float32)
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = (depth * 255).astype(np.uint8)
    # return cv.applyColorMap(depth, cv.COLORMAP_JET)
    return depth


def visualize_keypoints(keypoints: np.ndarray, image: np.ndarray, **kargs):
    """
    Visualize keypoints on image.
    
    Args:
        keypoints: Keypoints in [N, 2].
        image: Image in [H, W, C].
        **kargs: Keyword arguments passed to cv.circle.
        
    Returns:
        Image with keypoints.
    """
    radius = kargs['radius'] if 'radius' in kargs else 3
    color = kargs['color'] if 'color' in kargs else (0, 0, 255)
    thickness = kargs['thickness'] if 'thickness' in kargs else -1
    
    if image.ndim == 2 or image.shape[2] == 1:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    image = image.astype(np.uint8)
    for i in range(len(keypoints)):
        keypoint = keypoints[i]
        cv.circle(image, (int(keypoint[0]), int(keypoint[1])), radius, color, thickness)
    return image


def visualize_matches(image1: np.ndarray, keypoints1: np.ndarray, image2: np.ndarray, keypoints2: np.ndarray, matches: np.ndarray, **kargs):
    """
    Visualize matches between two images.
    
    Args:
        image1: Image in [H, W, C].
        keypoints1: Keypoints in [N, 2].
        image2: Image in [H, W, C].
        keypoints2: Keypoints in [N, 2].
        matches: Matches in [N, 2].
        **kargs: Keyword arguments passed to cv.drawMatches.
        
    Returns:
        Image with matches.
    """
    if image1.ndim == 2 or image1.shape[2] == 1:
        image1 = cv.cvtColor(image1, cv.COLOR_GRAY2BGR)
    if image2.ndim == 2 or image2.shape[2] == 1:
        image2 = cv.cvtColor(image2, cv.COLOR_GRAY2BGR)
    image1 = image1.astype(np.uint8)
    image2 = image2.astype(np.uint8)
    return cv.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, **kargs)
