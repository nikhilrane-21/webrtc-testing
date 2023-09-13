"""Media streamings"""
import logging
from pathlib import Path
from typing import Dict, Optional, cast
import av
from aiortc.contrib.media import MediaPlayer
from streamlit_webrtc import WebRtcMode, WebRtcStreamerContext, webrtc_streamer
import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import visualization_utils as vis_utils
import streamlit as st
import tempfile
import cv2

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)


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


model_dir = "model"
frozen_model_path = os.path.join(model_dir, "frozen_inference_graph.pb")
if not os.path.exists(frozen_model_path):
    st.error("frozen_inference_graph.pb file is not exist in model directory")
    exit(-1)
graph = load_model(frozen_model_path)
category_index = {1: {'id': 1, 'name': 'hardhat'},
                  2: {'id': 2, 'name': 'vest'},
                  3: {'id': 3, 'name': 'person'}}

MEDIAFILES: Dict[str, Dict] = {
    "_": {"local_file_path": None, "type": "video"},
}

media_file_label = st.radio("_", tuple(MEDIAFILES.keys()))
media_file_info = MEDIAFILES[cast(str, media_file_label)]

if media_file_info["local_file_path"] is None:
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            media_file_info = {"local_file_path": tmp_file.name, "type": "video"}
    else:
        st.warning("Please upload a video.")
        st.stop()

def create_player():
    return MediaPlayer(str(media_file_info["local_file_path"]))


key = f"media-streaming-{media_file_label}"
ctx: Optional[WebRtcStreamerContext] = st.session_state.get(key)

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # Create TensorFlow session outside the loop
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
        img = frame.to_ndarray(format="bgr24")
        image_expanded = np.expand_dims(img, axis=0)
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
        return av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")


webrtc_streamer(
    key=key,
    mode=WebRtcMode.RECVONLY,
    media_stream_constraints={
        "video": media_file_info["type"] == "video",
        "audio": False,
    },
    player_factory=create_player,
    video_frame_callback=video_frame_callback
)
