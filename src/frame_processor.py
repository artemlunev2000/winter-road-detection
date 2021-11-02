from cv2 import watershed, findContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, line, LINE_8, circle, inRange, vconcat
from scipy.optimize import minimize

import numpy as np


def border(start, true_cont):
    dist_sum = 0
    for point in true_cont:
        dist_sum += (abs(start[0] + start[1]*point[0] - point[1] - 15))

    return dist_sum


def process_frame(frame, hood_ending, marker_image, found_labels):
    found_labels = sorted(found_labels)
    markers = np.int32(marker_image)

    if hood_ending:
        upper_hood_part = markers[0:hood_ending]
        hood_part = np.zeros((markers.shape[0] - hood_ending, markers.shape[1], 1), np.int32)
        hood_label = 0

        for label in range(1, 256):
            if label not in found_labels:
                found_labels.append(label)
                hood_label = label
                break

        hood_part = circle(
            hood_part,
            (int(hood_part.shape[1] / 2), int((markers.shape[0] - hood_ending)*2/3)),
            15,
            hood_label,
            -1
        )

        markers = vconcat([upper_hood_part, hood_part])
        markers = np.int32(markers)

    # for custom markers
    # markers = np.zeros((frame.shape[0], frame.shape[1], 1), np.int32)
    #
    # centre_coords = [(int(frame.shape[1] / 2), hood_ending),
    #                  (int(frame.shape[1] / 2), hood_ending + int((frame.shape[0] - hood_ending) / 2)),
    #                  (int(frame.shape[1] / 2), 20), (20, int(frame.shape[0] / 2)),
    #                  (frame.shape[1] - 20, int(frame.shape[0] / 2))]
    #
    # for i, centre in enumerate(centre_coords):
    #     markers = circle(markers, centre, 9, i + 1, -1)

    markers = watershed(frame, markers)

    # colored result representation
    # colors = []
    # for i in range(len(found_labels)):
    #     b = np.random.uniform(0, 256)
    #     g = np.random.uniform(0, 256)
    #     r = np.random.uniform(0, 256)
    #     colors.append((b, g, r))
    # print(f'found {found_labels}')
    # dst_colored = np.zeros((markers.shape[0], markers.shape[1], 3), np.uint8)
    # for i in range(markers.shape[0]):
    #     for j in range(markers.shape[1]):
    #         if markers[i][j] > 0:
    #             dst_colored[i][j] = colors[found_labels.index(markers[i][j])]
    #
    # imshow('dst_col', dst_colored)

    if hood_ending:
        index = markers[hood_ending][int(markers.shape[1]/2)]
    else:
        index = markers[markers.shape[0] - 10][int(markers.shape[1]/2)]
    dst = inRange(markers, int(index), int(index))

    true_contours, img = findContours(dst, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    true_contour_left = []
    true_contour_right = []

    for point in true_contours[0]:
        point = point[0]
        if point[0] < markers.shape[1]/3 and point[1] < (hood_ending if hood_ending else markers.shape[0] - 20):
            true_contour_left.append(point)
        elif point[0] > 2*markers.shape[1]/3 and point[1] < (hood_ending if hood_ending else markers.shape[0] - 20):
            true_contour_right.append(point)

    x = np.array([400, -0.2])
    res = minimize(border, x, true_contour_left[::5])
    x1 = res.x

    frame = line(
        frame,
        (0, int(res.x[0])),
        (400, int(res.x[0] + res.x[1] * 400)),
        (100, 100, 200),
        5,
        LINE_8
    )

    x = np.array([400, -0.2])
    res = minimize(border, x, true_contour_right[::5])
    x2 = res.x

    frame = line(
        frame,
        (frame.shape[1], int(res.x[0] + res.x[1] * frame.shape[1])),
        (frame.shape[1] - 400, int(res.x[0] + res.x[1] * (frame.shape[1] - 400))),
        (100, 100, 200),
        5,
        LINE_8
    )

    # to see contour points on frame
    # for point in true_contour_left:
    #     frame[point[1]][point[0]] = [250, 250, 250]
    #
    # for point in true_contour_right:
    #     frame[point[1]][point[0]] = [250, 250, 250]

    return frame, x1, x2
