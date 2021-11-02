from cv2 import inRange, connectedComponentsWithStats, cvtColor, COLOR_BGR2GRAY, threshold, THRESH_BINARY, \
    erode, dilate, bitwise_and, findContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, vconcat
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


def find_horizon(road_frame):
    contours, img = findContours(road_frame, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    contours = [point[0] for point in contours[0]]
    contours = sorted(contours, key=lambda x: x[1])

    number_of_pixels_in_slice = 5
    min_thickness = road_frame.shape[1]
    min_thickness_coordinate = None

    current_left_pixel = road_frame.shape[1]
    current_right_pixel = 0
    current_slice_start = contours[0][1]
    for point in contours:
        if point[1] - current_slice_start >= number_of_pixels_in_slice:
            current_thickness = current_right_pixel - current_left_pixel
            middle = (current_left_pixel + current_right_pixel) / 2
            if road_frame.shape[1] / 20 < current_thickness < min_thickness and \
                    road_frame.shape[1] / 3 < middle < road_frame.shape[1] * 2 / 3:
                min_thickness = current_thickness
                min_thickness_coordinate = current_slice_start + int(number_of_pixels_in_slice / 2)
            current_slice_start = point[1]
            current_left_pixel = road_frame.shape[1]
            current_right_pixel = 0

        if point[0] < current_left_pixel:
            current_left_pixel = point[0]
        if point[0] > current_right_pixel:
            current_right_pixel = point[0]

    lower_part = road_frame[min_thickness_coordinate:road_frame.shape[0]]
    upper_part = np.zeros((road_frame.shape[0] - lower_part.shape[0], road_frame.shape[1], 1), road_frame.dtype)
    road_frame = vconcat([upper_part, lower_part])

    return road_frame


def preprocess_frame(frame):
    frame = cvtColor(frame, COLOR_BGR2GRAY)
    threshold(frame, 70, 255, THRESH_BINARY, frame)

    # get rid of noise
    kernel = np.ones((7, 7), np.uint8)
    frame = erode(frame, kernel)
    frame = dilate(frame, kernel)

    retval, labels, stats, centroids = connectedComponentsWithStats(frame, connectivity=4)
    labels = np.uint8(labels)

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
    components = analyse_components(components, frame)

    found_labels = [comp['label'] + 1 for comp in components if comp is not None]
    labels = labels + 1

    markers_images = {lbl: bitwise_and(labels, inRange(labels, lbl, lbl)) for lbl in found_labels}
    markers_images[components[1]['label']+1] = find_horizon(markers_images[components[1]['label']+1])

    erode_final_kernel = np.ones((80, 80), np.uint8)
    markers_images = {k: erode(v, erode_final_kernel) for k, v in markers_images.items()}

    final_marker_image = sum(markers_images.values())

    # visualize markers
    # for i, im in enumerate(markers_images.values()):
    #     imshow(f'{i}', im*20)

    return final_marker_image, found_labels
