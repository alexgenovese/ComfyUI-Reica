import numpy as np
import cv2


def get_bbox_from_mask(mask):
    """Get bounding box from mask"""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return [0, 0, 0, 0]
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    return [rmin, rmax + 1, cmin, cmax + 1]


def expand_bbox(image, bbox, ratio):
    """Expand bounding box by ratio"""
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
    else:
        h, w = image
    
    y1, y2, x1, x2 = bbox
    
    center_y = (y1 + y2) / 2
    center_x = (x1 + x2) / 2
    
    height = y2 - y1
    width = x2 - x1
    
    new_height = height * ratio
    new_width = width * ratio
    
    new_y1 = max(0, int(center_y - new_height / 2))
    new_y2 = min(h, int(center_y + new_height / 2))
    new_x1 = max(0, int(center_x - new_width / 2))
    new_x2 = min(w, int(center_x + new_width / 2))
    
    return [new_y1, new_y2, new_x1, new_x2]


def pad_to_square(image, pad_value=0, random=False):
    """Pad image to square"""
    if len(image.shape) == 3:
        h, w, c = image.shape
    else:
        h, w = image.shape
        c = 1
    
    if h == w:
        return image
    
    max_dim = max(h, w)
    
    if len(image.shape) == 3:
        padded = np.full((max_dim, max_dim, c), pad_value, dtype=image.dtype)
    else:
        padded = np.full((max_dim, max_dim), pad_value, dtype=image.dtype)
    
    start_y = (max_dim - h) // 2
    start_x = (max_dim - w) // 2
    
    if len(image.shape) == 3:
        padded[start_y:start_y + h, start_x:start_x + w, :] = image
    else:
        padded[start_y:start_y + h, start_x:start_x + w] = image
    
    return padded


def box2squre(image, bbox):
    """Convert bounding box to square"""
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
    else:
        h, w = image
    
    y1, y2, x1, x2 = bbox
    
    center_y = (y1 + y2) / 2
    center_x = (x1 + x2) / 2
    
    height = y2 - y1
    width = x2 - x1
    
    size = max(height, width)
    
    new_y1 = max(0, int(center_y - size / 2))
    new_y2 = min(h, int(center_y + size / 2))
    new_x1 = max(0, int(center_x - size / 2))
    new_x2 = min(w, int(center_x + size / 2))
    
    return [new_y1, new_y2, new_x1, new_x2]


def crop_back(edited_image, original_image, dims, crop_bbox):
    """Crop back the edited image to original size"""
    H1, W1, H2, W2 = dims
    y1, y2, x1, x2 = crop_bbox
    
    # Resize edited image back to cropped size
    edited_resized = cv2.resize(edited_image, (H1, W1))
    
    # Create result image with original size
    result = original_image.copy()
    
    # Place edited region back
    result[y1:y2, x1:x2] = edited_resized
    
    return result


def expand_image_mask(image, mask, ratio=1.3):
    """Expand image and mask by padding"""
    h, w = image.shape[:2]
    
    new_h = int(h * ratio)
    new_w = int(w * ratio)
    
    # Calculate padding
    pad_h = (new_h - h) // 2
    pad_w = (new_w - w) // 2
    
    # Pad image
    if len(image.shape) == 3:
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), 
                             mode='constant', constant_values=255)
    else:
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), 
                             mode='constant', constant_values=255)
    
    # Pad mask
    padded_mask = np.pad(mask, ((pad_h, pad_h), (pad_w, pad_w)), 
                        mode='constant', constant_values=0)
    
    return padded_image, padded_mask