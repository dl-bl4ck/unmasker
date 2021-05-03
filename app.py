import streamlit as st 
from PIL import Image
from predict import predict

st.title("Unmasker")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    output = predict('model_cn_step3',image)
    st.image(image, caption='Uploaded Image.')
    st.image(output,caption='Generated Image')
 