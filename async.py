import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import visualization_utils as vis_utils
import streamlit as st
from PIL import Image
import tempfile
import cv2
import asyncio

def load_model(inference_model_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(inference_model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, sess, tensor_dict):
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})

    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    return output_dict


async def process_frame(frame, sess, tensor_dict):
    img = np.array(frame)  # Convert image to NumPy array
    image_expanded = np.expand_dims(frame, axis=0)
    output_dict = run_inference_for_single_image(image_expanded, sess, tensor_dict)
    vis_utils.visualize_boxes_and_labels_on_image_array(
        img,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=4)
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return frame_rgb

async def display_frame(frame_container, frame):
    frame_container.image(frame, channels='RGB')

frozen_model_path = os.path.join("model", "frozen_inference_graph.pb")
if not os.path.exists(frozen_model_path):
    st.error("frozen_inference_graph.pb file is not exist in model directory")
    exit(-1)
graph = load_model(frozen_model_path)
category_index = {1: {'id': 1, 'name': 'hardhat'},
                  2: {'id': 2, 'name': 'vest'},
                  3: {'id': 3, 'name': 'person'}}




uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mkv", "mov"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file.seek(0)

        if st.button("Submit"):
            cap = cv2.VideoCapture(temp_file.name)

            frame_container = st.empty()
            with tf.Session(graph=graph) as sess:
                ops = graph.get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = graph.get_tensor_by_name(tensor_name)
                while True:
                    ret, frame = cap.read()  # Read a frame from the video file
                    if not ret:
                        break
                    processed_frame = asyncio.run(process_frame(frame, sess, tensor_dict))
                    asyncio.run(display_frame(frame_container, processed_frame))
                cap.release()

