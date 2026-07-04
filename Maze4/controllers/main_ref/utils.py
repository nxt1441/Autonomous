import numpy as np
import cv2
from CONSTANTS import *


def angle_wrap(a, b):
    diff = a - b
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff < -np.pi:
        diff += 2 * np.pi
    return diff


def drop_single_pixels(grid_map, obstacle_value=1, connectivity=8):
    binary = np.where(grid_map == obstacle_value, 255, 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=connectivity)
    out = np.zeros_like(binary)
    for lbl in range(1, n):
        if stats[lbl, cv2.CC_STAT_AREA] > 1:
            out[labels == lbl] = 255
    return np.where(out == 255, obstacle_value, 0)


def remove_small_blobs(grid_map, obstacle_value=1, min_size=5, connectivity=8):
    binary = np.where(grid_map == obstacle_value, 255, 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=connectivity)
    keep = np.zeros_like(binary)
    for lbl in range(1, n):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_size:
            keep[labels == lbl] = 255
    result = grid_map.copy()
    result[(keep == 0) & (grid_map == obstacle_value)] = 0
    return result


def dilate_obstacles(grid_map, inflation_pixels=10):
    ksize = int(2 * inflation_pixels + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    binary = np.where(grid_map == 1, 255, 0).astype(np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    return (dilated > 0).astype(np.uint8)


def clear_around_point(map_array, map_point, inflation_pixels=1):
    h, w = map_array.shape
    x, y = map_point[0], map_point[1]
    r2 = inflation_pixels * inflation_pixels
    for dy in range(-inflation_pixels, inflation_pixels + 1):
        for dx in range(-inflation_pixels, inflation_pixels + 1):
            if dx * dx + dy * dy <= r2:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    map_array[ny, nx] = 0


def ray_cells(start, end):
    x1, y1 = start
    x2, y2 = end
    pts = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    while True:
        pts.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return pts


def extract_color_mask(hsv_img, color):
    if color == 'red':
        m1 = cv2.inRange(hsv_img, np.array(RED_WALL_HSV_LOWER1), np.array(RED_WALL_HSV_UPPER1))
        m2 = cv2.inRange(hsv_img, np.array(RED_WALL_HSV_LOWER2), np.array(RED_WALL_HSV_UPPER2))
        return cv2.bitwise_or(m1, m2)
    if color == 'blue':
        return cv2.inRange(hsv_img, np.array(BLUE_HSV_LOWER), np.array(BLUE_HSV_UPPER))
    if color == 'yellow':
        return cv2.inRange(hsv_img, np.array(YELLOW_HSV_LOWER), np.array(YELLOW_HSV_UPPER))
    if color == 'green':
        return cv2.inRange(hsv_img, np.array(GREEN_HSV_LOWER), np.array(GREEN_HSV_UPPER))
    return None


def red_wall_centered(hsv_img):
    mask = extract_color_mask(hsv_img, 'red')
    if cv2.countNonZero(mask) == 0:
        return False, mask
    h, w, _ = hsv_img.shape
    border = 30
    if (np.any(mask[:border, :]) or np.any(mask[h - border:, :]) or
            np.any(mask[:, :border]) or np.any(mask[:, w - border:])):
        return False, mask
    return True, mask


def map_delta_ratio(map1, map2):
    return np.sum(map1 != map2) / (map1.shape[0] * map1.shape[1])
