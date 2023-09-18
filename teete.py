import cv2
import streamlit as st
import tempfile
import base64


def generate_frames(video):
    while True:
        success, frame = video.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        yield base64.b64encode(frame).decode()


if __name__ == "__main__":
    st.title("Video Player")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mkv", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        vf = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        for frame in generate_frames(vf):
            # Display the frame in an HTML img element
            stframe.markdown(f'<img src="data:image/jpeg;base64,{frame}"/>', unsafe_allow_html=True)

        vf.release()
