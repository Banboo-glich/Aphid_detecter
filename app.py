import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
import detect as detect


st.set_page_config(page_title="AphidDetector",
                   layout='wide')


def train_models():
    detect.train()
    print("[INFO] Training Detection model done!") 
    
    
def main():
    
    st.sidebar.title("Settings")
    st.sidebar.subheader("Parameters")
    st.markdown(
    """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width:300px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width:300px;
            margin-left:-300px;
        }
        </style>
    """,
    unsafe_allow_html=True,
    )
    
    app_mode = st.sidebar.selectbox('Choose the App Mode', ['アプリについて', '画像解析'])
    
    if app_mode == 'Introduction':

        st.title('深層学習(AI)によるアブラムシの画像解析 ')
        st.header("この子はいくつ？")
        image = Image.open('1-G1-1028-13.jpg')
        st.image(image, caption='-エンドウヒゲナガアブラムシ(PeaAphid)-')



    elif app_mode == "Object Detection":
        
        st.header("Classification of developmental stages in pea aphid using YOLOv8 ",)
        
        st.sidebar.markdown("----")
        confidence = st.sidebar.slider("Confidence", min_value=0.0, max_value=1.0, value=0.50)
        
        img_file_buffer_detect = st.sidebar.file_uploader("Upload an image", type=['jpg','jpeg', 'png'], key=0)
        DEMO_IMAGE = "1-G1-1028-13.jpg"
        
        if img_file_buffer_detect is not None:
            img = cv.imdecode(np.frombuffer(img_file_buffer_detect.read(), np.uint8), 1)
            image = np.array(Image.open(img_file_buffer_detect))
        else:
            img = cv.imread(DEMO_IMAGE)
            image = np.array(Image.open(DEMO_IMAGE))
        st.sidebar.text("Original Image")
        st.sidebar.image(image)
        
        # predict
        detect.predict(img, confidence, st)
        st.balloons()
        
 

if __name__ == "__main__":
    try:
        
        # RUN THE FOLLOWING ONLY IF YOU WANT TO TRAIN MODEL AGAIN 
        # train_models()
        
        main()
    except SystemExit:
        pass

