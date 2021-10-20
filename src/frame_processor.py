from cv2 import watershed, findContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, line, LINE_8, circle
from scipy.optimize import minimize

import numpy as np


def border(start, true_cont):
    sum = 0
    for point in true_cont:
        sum += (abs(start[0] + start[1]*point[0] - point[1] - 15))

    return sum


def process_frame(frame, hood_ending, marker_image):
    s = time()
    markers = np.int32(marker_image)
    e = time()
    print(f'1 - {e - s}')

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
    s = time()
    markers = watershed(frame, markers)
    e = time()
    print(f'2 - {e - s}')
    # for colored result

    # colors = []
    # for i in range(5):
    #     b = np.random.uniform(0, 256)
    #     g = np.random.uniform(0, 256)
    #     r = np.random.uniform(0, 256)
    #     colors.append((b, g, r))
    #
    # dst_colored = np.zeros((markers.shape[0], markers.shape[1], 3), np.uint8)
    # for i in range(markers.shape[0]):
    #     for j in range(markers.shape[1]):
    #         if markers[i][j] > 0:
    #             dst_colored[i][j] = colors[markers[i][j][0]-1]
    #
    # imshow('dst_col', dst_colored)

    index = markers[hood_ending][int(markers.shape[1]/2)]
    s = time()
    dst = np.zeros(markers.shape, np.uint8)

    for i in range(markers.shape[0]):
        for j in range(markers.shape[1]):
            if markers[i][j] == index:
                dst[i][j] = 250
    e = time()
    print(f'3 - {e - s}')
    s = time()
    true_contours, img = findContours(dst, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    true_contour1 = []
    true_contour2 = []

    for point in true_contours[0]:
        point = point[0]
        if point[0] < markers.shape[1]/3:
            true_contour1.append(point)
        elif point[0] > 2*markers.shape[1]/3:
            true_contour2.append(point)
    e = time()
    print(f'4 - {e - s}')

    x = np.array([300, 0.2])
    s = time()
    res = minimize(border, x, true_contour1)
    x1 = res.x

    frame = line(frame,
                 (0, int(res.x[0])),
                 (400, int(res.x[0] + res.x[1] * 400)),
                 (100, 100, 200),
                 5,
                 LINE_8)

    x = np.array([300, -0.2])

    res = minimize(border, x, true_contour2)
    x2 = res.x

    frame = line(frame,
                 (frame.shape[1], int(res.x[0] + res.x[1] * frame.shape[1])),
                 (frame.shape[1] - 400, int(res.x[0] + res.x[1] * (frame.shape[1] - 400))),
                 (100, 100, 200),
                 5,
                 LINE_8)
    e = time()
    print(f'5 - {e - s}')

    # frame = np.zeros(frame.shape, np.uint8)
    # for point in true_contour1:
    #     frame[point[1]][point[0]] = [250, 250, 250]
    #
    # for point in true_contour2:
    #     frame[point[1]][point[0]] = [250, 250, 250]

    return frame, x1, x2


if __name__ == '__main__':
    from cv2 import imread, imshow, waitKey, destroyAllWindows
    from src.frame_preprocessor import preprocess_frame
    from time import time

    fr = imread("../orig.png")
    s = time()
    markers_image, horiz = preprocess_frame(fr)
    e = time()
    print(e-s)
    s = time()
    fr, _, _ = process_frame(fr, 569, markers_image)
    e = time()
    print(e - s)
    imshow('res', fr)
    waitKey(0)
    destroyAllWindows()