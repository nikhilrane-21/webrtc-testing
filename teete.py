import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import visualization_utils as vis_utils
import streamlit as st
from PIL import Image
import tempfile
import cv2
import base64

st.set_page_config(page_title="Streamlit Helmet Vest Hardhat Detection Demo", page_icon="🤖")
hide_streamlit_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

task_list = ["Video", "RTSP", "Image"]

with st.sidebar:
    st.title('Source Selection')
    task_name = st.selectbox("Select your source type:", task_list)
st.title(task_name)

def load_model(inference_model_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(inference_model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def run_inference_for_single_image(image, sess, tensor_dict):
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    return output_dict

def image_processing(graph, category_index, image):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_filename = temp_file.name
        cv2.imwrite(temp_filename, image)
        img = cv2.imread(temp_filename)
        image_expanded = np.expand_dims(img, axis=0)
        with graph.as_default():
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            with tf.Session() as sess:
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
                return img

def webcam_processing(category_index, frame, sess, tensor_dict):
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
    _, buffer = cv2.imencode('.jpg', img)
    frame_base64 = base64.b64encode(buffer).decode()
    return frame_base64

model_dir = "model"
frozen_model_path = os.path.join(model_dir, "frozen_inference_graph.pb")
if not os.path.exists(frozen_model_path):
    st.error("frozen_inference_graph.pb file is not exist in model directory")
    exit(-1)
graph = load_model(frozen_model_path)
category_index = {1: {'id': 1, 'name': 'hardhat'},
                  2: {'id': 2, 'name': 'vest'},
                  3: {'id': 3, 'name': 'person'}}

if task_name == task_list[0]:
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
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_base64 = webcam_processing(category_index, frame, sess, tensor_dict)
                        frame_container.markdown(f'<img src="data:image/jpeg;base64,{frame_base64}"/>', unsafe_allow_html=True)
                cap.release()

if task_name == task_list[1]:
    rtsp_link = st.text_input("Enter the RTSP link")
    if rtsp_link:
        cap = cv2.VideoCapture(rtsp_link)
        if st.button("Submit"):
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
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_base64 = webcam_processing(category_index, frame, sess, tensor_dict)
                    frame_container.markdown(f'<img src="data:image/jpeg;base64,{frame_base64}"/>', unsafe_allow_html=True)
                cap.release()

if task_name == task_list[2]:
    uploaded_file = st.file_uploader("Choose a image file", type=["png", "jpeg", "jpg", "tif"])
    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        image = np.array(pil_image)
        image = image_processing(graph, category_index, image)
        st.subheader("Output:")
        st.image(image)

# if task_name == task_list[3]:
#     cap = cv2.VideoCapture(0)
#     if st.button("Start"):
#         frame_container = st.empty()
#         with tf.Session(graph=graph) as sess:
#             ops = graph.get_operations()
#             all_tensor_names = {output.name for op in ops for output in op.outputs}
#             tensor_dict = {}
#             for key in [
#                 'num_detections', 'detection_boxes', 'detection_scores',
#                 'detection_classes', 'detection_masks'
#             ]:
#                 tensor_name = key + ':0'
#                 if tensor_name in all_tensor_names:
#                     tensor_dict[key] = graph.get_tensor_by_name(tensor_name)
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#                 frame_base64 = webcam_processing(category_index, frame, sess, tensor_dict) 
#                 frame_container.markdown(f'<img src="data:image/jpeg;base64,{frame_base64}"/>', unsafe_allow_html=True)
#         cap.release()
