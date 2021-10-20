from cv2 import inRange, connectedComponentsWithStats, cvtColor, COLOR_BGR2GRAY, threshold, THRESH_BINARY, \
    erode, dilate, bitwise_and
from time import time

import numpy as np


def analyse_components(components, frame):
    road = sky = left = right = None

    road_sky_components = sorted(components[:2], key=lambda x: x['center'][1])
    sky, road = road_sky_components[0], road_sky_components[1]

    left_right_components = sorted(components[2:4], key=lambda x: x['center'][0])
    if left_right_components[0]['center'][0] < frame.shape[1] / 3:
        left = left_right_components[0]
    if left_right_components[1]['center'][0] > frame.shape[1] * 2 / 3:
        right = left_right_components[1]

    return [sky, road, left, right]


def find_horizon(road_component, frame):
    number_of_pixels_in_slice = 6
    min_pixels_sum = None
    horizon = None

    for y_coord in range(int(frame.shape[0] / 3), int(frame.shape[0] * 2 / 3), number_of_pixels_in_slice):
        pixels_sum = 0
        for x_coord in range(frame.shape[1]):
            for i in range(number_of_pixels_in_slice):
                if frame[y_coord+i][x_coord] == road_component['label']:
                    pixels_sum += 1
        if (min_pixels_sum is None or pixels_sum < min_pixels_sum) and pixels_sum != 0:
            min_pixels_sum = pixels_sum
            horizon = y_coord + number_of_pixels_in_slice / 2

    return int(horizon + 50)


def preprocess_frame(frame):
    s = time()
    frame = cvtColor(frame, COLOR_BGR2GRAY)
    threshold(frame, 70, 255, THRESH_BINARY, frame)
    e = time()
    print(f'6 - {e - s}')
    # get rid of noise
    s = time()
    kernel = np.ones((7, 7), np.uint8)
    frame = erode(frame, kernel)
    frame = dilate(frame, kernel)
    e = time()
    print(f'7 - {e - s}')
    s = time()
    retval, labels, stats, centroids = connectedComponentsWithStats(frame, connectivity=8)
    e = time()
    print(f'8 - {e - s}')
    labels = np.uint8(labels)

    s = time()
    components_sizes = stats[0:, -1]
    components = [
        {
            'center': centroids[i],
            'size': components_sizes[i],
            'label': i
        }
        for i in range(retval)
    ]
    components = sorted(components, key=lambda x: x['size'], reverse=True)[:4]
    e = time()
    print(f'9 - {e - s}')

    s = time()
    components = analyse_components(components, frame)
    e = time()
    print(f'10 - {e - s}')
    s = time()
    horizon = find_horizon(components[1], labels)
    e = time()
    print(f'11 - {e - s}')
    found_labels = [comp['label'] for comp in components if comp is not None]

    erode_final_kernel = np.ones((15, 15), np.uint8)
    s = time()
    markers_images = [bitwise_and(labels + 1, inRange(labels+1, lbl+1, lbl+1)) for lbl in found_labels]
    e = time()
    print(f'12 - {e - s}')
    s = time()
    final_marker_image = sum(markers_images)
    e = time()
    print(f'13 - {e - s}')
    return final_marker_image, horizon
