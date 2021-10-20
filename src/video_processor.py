from cv2 import waitKey, destroyAllWindows, line, LINE_8, bitwise_and, cvtColor, COLOR_BGR2GRAY, \
    threshold, THRESH_BINARY, dilate

import numpy as np
import cv2 as cv

from src.frame_processor import process_frame
from src.frame_preprocessor import preprocess_frame


def detect_hood_ending(images):
    result = bitwise_and(images[0], images[1])
    for image in images[2:]:
        result = bitwise_and(result, image)

    threshold(result, 70, 255, THRESH_BINARY, result)
    kernel = np.ones((7, 7), np.uint8)
    result = dilate(result, kernel)

    current_white_pixels = 0
    for height in range(result.shape[0] - 1, 0, -1):
        if result[height][int(result.shape[1]/2)] == 0:
            current_white_pixels = 0
        else:
            current_white_pixels += 1
            if current_white_pixels == 10:
                return result.shape[0] - height

    return 0


def process_video(path):
    cap = cv.VideoCapture(path)
    frame_counter = 0
    x1 = x2 = None

    possible_hood_area_images = []
    needed_hood_area_images = 40
    hood_ending = None

    while cap.isOpened():
        frame_counter += 1
        ret, frame = cap.read()
        new_frame = frame[39:frame.shape[0], 0:frame.shape[1]]

        if hood_ending is None:
            if len(possible_hood_area_images) < needed_hood_area_images:
                possible_hood_area_images.append(
                    cvtColor(frame[int(frame.shape[0]*2/3):frame.shape[0], 0:frame.shape[1]], COLOR_BGR2GRAY)
                )
            else:
                hood_ending = new_frame.shape[0] - detect_hood_ending(possible_hood_area_images) - 40

        if not ret:
            print("Can't receive frame. Exiting.")
            break

        if frame_counter % 100 == 0:
            markers_image, horizon = preprocess_frame(new_frame)
            new_frame, x1, x2 = process_frame(new_frame, hood_ending, markers_image)
            cv.imshow('frame', new_frame)
            frame_counter = 0
        elif x1 is not None and x2 is not None:
            new_frame = line(new_frame,
                             (0, int(x1[0])),
                             (400, int(x1[0] + x1[1] * 400)),
                             (50, 50, 50),
                             5,
                             LINE_8)
            new_frame = line(new_frame,
                             (new_frame.shape[1], int(x2[0] + x2[1] * new_frame.shape[1])),
                             (new_frame.shape[1] - 400, int(x2[0] + x2[1] * (new_frame.shape[1] - 400))),
                             (50, 50, 50),
                             5,
                             LINE_8)
            cv.imshow('frame', new_frame)
        else:
            cv.imshow('frame', new_frame)
        if cv.waitKey(1) == ord('q'):
            break

    waitKey(0)
    destroyAllWindows()
    cap.release()
    cv.destroyAllWindows()
