
# You can any WASD games like pacman, subway surfers, temple run etc.
# The code is inspired by https://github.com/uvipen/AirGesture

import tensorflow as tf
import cv2
import numpy as np
import multiprocessing as _mp
from utils import load_graph, detect_hands, predict, is_in_triangle
from utils import RED, CYAN, YELLOW, BLUE, GREEN
from pyKey import pressKey, releaseKey, press #inspired by https://github.com/andohuman/pyKey for controlling keyboard keys
import keyboard

width = 640
height = 480
threshold = 0.6
alpha = 0.3
pre_trained_model_path = "model/pretrained_model.pb"

def main():
    graph, sess = load_graph(pre_trained_model_path)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    mp = _mp.get_context("spawn")
    v = mp.Value('i', 0)
    lock = mp.Lock()

    x_center = int(width / 2)
    y_center = int(height / 2)
    radius = int(min(width, height) / 12)
    while True:
        key = cv2.waitKey(10)
        if key == ord("q"):
            break
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, scores, classes = detect_hands(frame, graph, sess)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = predict(boxes, scores, classes, threshold, width, height)
        if len(results) == 1:
            x_min, x_max, y_min, y_max, category = results[0]
            x = int((x_min + x_max) / 2)
            y = int((y_min + y_max) / 2)
            cv2.circle(frame, (x, y), 5, RED, -1)
            
            if category == "Open" and np.linalg.norm((x - x_center, y - y_center)) <= radius:
                action = 0 # Stay
                text = "Stay"
            
            elif category == "Open" and is_in_triangle((x, y), [(0, 0), (width, 0),(x_center, y_center)]):
                action = 1  # Up
                text = "Up"
                keyboard.press_and_release('up')
                
                
            elif category == "Open" and is_in_triangle((x, y), [(0, height), (width, height), (x_center, y_center)]):
                action = 2  # Down
                text = "Down"
                keyboard.press_and_release('down')
                
            elif category == "Open" and is_in_triangle((x, y), [(0, 0), (0, height), (x_center, y_center)]):
                action = 3  # Left
                text = "Left"
                keyboard.press_and_release('left')
                
            elif category == "Open" and is_in_triangle((x, y), [(width, 0), (width, height), (x_center, y_center)]):
                action = 4  # Right
                text = "Right"
                keyboard.press_and_release('right')
            
            else:
                action = 0
                text = "Stay"
                
            with lock:
                v.value = action
            cv2.putText(frame, "{}".format(text), (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)

        overlay = frame.copy()
        cv2.drawContours(overlay, [np.array([(0, 0), (width, 0), (x_center, y_center)])], 0, CYAN, -1)
        cv2.drawContours(overlay, [np.array([(0, height), (width, height), (x_center, y_center)])], 0, CYAN, -1)
        cv2.drawContours(overlay, [np.array([(0, 0), (0, height), (x_center, y_center)])], 0, YELLOW, -1)
        cv2.drawContours(overlay, [np.array([(width, 0), (width, height), (x_center, y_center)])], 0, YELLOW, -1)
        cv2.circle(overlay, (x_center, y_center), radius, BLUE, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.imshow('Detection', frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

