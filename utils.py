import numpy as np
import gym
import tensorflow as tf
from time import sleep

IS_FULLSCREEN = False
WIDTH = 520
HEIGHT = 600
HAND_GESTURES = ["Open", "Closed"]
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
ORANGE = (0,128,255)
YELLOW = (0,255,255)
MAGENTA = (255,0,255)
CYAN = (255,255,0)
PURPLE = (128,0,128)


def is_in_triangle(point, triangle):
    # barycentric coordinate system
    x, y = point
    (xa, ya), (xb, yb), (xc, yc) = triangle
    a = ((yb - yc)*(x - xc) + (xc - xb)*(y - yc)) / ((yb - yc)*(xa - xc) + (xc - xb)*(ya - yc))
    b = ((yc - ya) * (x - xc) + (xa - xc) * (y - yc)) / ((yb - yc) * (xa - xc) + (xc - xb) * (ya - yc))
    c = 1 - a - b
    if 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1:
        return True
    else:
        return False


def load_graph(path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(path, 'rb') as fid:
            graph_def.ParseFromString(fid.read())
            tf.import_graph_def(graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    return detection_graph, sess


def detect_hands(image, graph, sess):
    input_image = graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = graph.get_tensor_by_name('detection_scores:0')
    detection_classes = graph.get_tensor_by_name('detection_classes:0')
    image = image[None, :, :, :]
    boxes, scores, classes = sess.run([detection_boxes, detection_scores, detection_classes],
                                      feed_dict={input_image: image})
    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)


def predict(boxes, scores, classes, threshold, width, height, num_hands=2):
    count = 0
    results = {}
    for box, score, class_ in zip(boxes[:num_hands], scores[:num_hands], classes[:num_hands]):
        if score > threshold:
            y_min = int(box[0] * height)
            x_min = int(box[1] * width)
            y_max = int(box[2] * height)
            x_max = int(box[3] * width)
            category = HAND_GESTURES[int(class_) - 1]
            results[count] = [x_min, x_max, y_min, y_max, category]
            count += 1
    return results


