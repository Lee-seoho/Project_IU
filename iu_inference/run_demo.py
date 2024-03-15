import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import tensorflow as tf
import argparse
import numpy as np
from PIL import Image
from imutils import paths
import matplotlib.pyplot as plt
import os
import cv2
import random

# Tensorboard 보는 법
# tensorboard --logdir ./
# checkpoint 쌓이는 경로 입력하면 됨.
# 터미널 새로 켜서 확인해야 함

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

if __name__ == "__main__":
  # parser
  parser = argparse.ArgumentParser()
  parser.add_argument('-s', '--saved_model', 
                      required=True, default='models/240304_saved_model', 
                      help='path to saved model')
  parser.add_argument('-i', '--image_directory', 
                      required=True, default='dataset/test', 
                      help='test image directory')
  parser.add_argument('-o', '--output', 
                      required=False, default='output', 
                      help='output directory name')
  parser.add_argument('-c', '--confidence',
                      required=False, default=0.8, 
                      help='confidence threshold')
  args = parser.parse_args()

  # load detection model
  # 1. saved_model 폴더
  # 2. label.pbtxt
  detect_fn = tf.saved_model.load(args.saved_model)
  category_index = label_map_util.create_category_index_from_labelmap(
                      args.saved_model+"/label_map.pbtxt",
                      use_display_name=True)
  
  # output directory
  os.makedirs(args.output, exist_ok=True)
  
  for img_path in sorted(list(paths.list_images(args.image_directory))):
    print("img_path:", img_path)

    image_np = load_image_into_numpy_array(img_path)
    
    input_tensor = tf.convert_to_tensor(image_np) # 형변환
    input_tensor = input_tensor[tf.newaxis, ...] # 차원으로 만들기
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # class processing
    detections['detection_classes'] = detections[
                                        'detection_classes'
                                      ].astype(np.int64)
    image_np_with_detections = image_np.copy()
    image_np_with_detections = cv2.cvtColor(image_np_with_detections, 
                                            cv2.COLOR_RGB2BGR)

    # visualization
    boxes = detections['detection_boxes']
    classes = detections['detection_classes']
    scores = detections['detection_scores']

    for box, label, score in zip(boxes, classes, scores):
        if score < args.confidence:
          continue

        y1, x1, y2, x2 = box
        x1 = int(x1 * image_np.shape[1])
        y1 = int(y1 * image_np.shape[0])
        x2 = int(x2 * image_np.shape[1])
        y2 = int(y2 * image_np.shape[0])

        # Set box color based on class
        if label == 1:
            box_color = (0, 0, 255)  # Red for class 1
        elif label == 2:
            box_color = (0, 255, 0)  # Green for class 2
        elif label == 3:
            box_color = (255, 0, 0)  # Blue for class 3
        else:
            box_color = (0, 255, 255)  # Yellow for class 4

        print("box:", x1, y1, x2, y2)
        cv2.rectangle(image_np_with_detections, (x1, y1), (x2, y2), box_color, 2)
        # cv2.putText()
        # 1. 텍스트 띄워주는 것 + score 값 표시
        # 2. label 별로 색 다르게 해서 표시해보기
        label_text = f'{category_index[label]["name"]}: {int(score*100)}%'
        cv2.putText(image_np_with_detections, label_text, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("window", image_np_with_detections)
    key = cv2.waitKey(0) & 0xff
    if key == ord('q'):
       break
    
    time.sleep(0.5) # image

  print("done")