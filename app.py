
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

st.set_page_config(
    page_title="AI Fire Smoke Detection",
    page_icon="🔥"
)

st.title("🔥 AI Fire & Smoke Detection System")

MODEL_PATH = "/content/drive/MyDrive/YOLO_Object_Detection/models/Fire_Smoke_YOLOv11_Final_Fixed/weights/best.pt"


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()


uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file:

    # THIS CREATES IMAGE VARIABLE
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original")
    st.image(image)


    # Save image
    path = "/content/test.jpg"

    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())


    st.write("File exists:", os.path.exists(path))


    # Prediction
    results = model.predict(
        source=path,
        conf=0.25,
        imgsz=640,
        verbose=False
    )


    result = results[0]


    st.subheader("Result")
    st.image(result.plot())


    if len(result.boxes) == 0:

        st.success("No detection")


    else:

        for box in result.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            name = model.names[cls]

            st.write(
                f"{name} : {conf:.3f}"
            )


        best = int(result.boxes.conf.argmax())

        cls = int(result.boxes.cls[best])
        conf = float(result.boxes.conf[best])

        if model.names[cls] == "fire":
            st.error(f"🔥 FIRE {conf:.2f}")

        elif model.names[cls] == "smoke":
            st.warning(f"🌫 SMOKE {conf:.2f}")

        else:
            st.info("OTHER")
