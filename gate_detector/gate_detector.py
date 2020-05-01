
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

MODEL_NAME = 'inference_graph'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

NUM_CLASSES = 1

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    sess = tf.Session(graph=detection_graph, config=config)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

class gateDetector(object):
    def __init__(self, MODEL_NAME, CWD_PATH, PATH_TO_CKPT, PATH_TO_LABELS, NUM_CLASSES):
        ## Load the label map.
        self.label_map = self.load_label_map(PATH_TO_LABELS)
        self.categories = self.load_categories(NUM_CLASSES)
        self.category_index = label_map_util.create_category_index(categories)

        ## Load the Tensorflow model into memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            config = tf.ConfigProto()
            # config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.7
            sess = tf.Session(graph=detection_graph, config=config)

    def __enter__(self):
        self.detection_graph = tf.Graph()
        return self.detection_graph

    def __exit__(self, type, value, traceback):
        self.detection_graph.close()

    def load_label_map(self, path_to_labels):
        return label_map_util.load_labelmap(path_to_labels)
    
    def load_categories(self, num_classes):
        return label_map_util.convert_label_map_to_categories(label_map, 
                                                            max_num_classes=num_classes,
                                                            use_display_name=True)

    

                                        
        
        

