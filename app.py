import streamlit as st
from PIL import Image
from model.caption_model import model,image_processor, device  

st.set_page_config(page_title="Image Captioning", layout="centered")
st.title("🖼️ Image Captioning using Deep Learning")

@st.cache_resource
def get_model():
    return model()

model = get_model()

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating caption..."):
        img = image_processor(image, return_tensors="pt").to(device)
        output = model.generate(**img)


    st.success("Generated Caption:")
    st.markdown(f"**📌 {output}**")
    
    
# def load_model(model, image_processor, tokenizer, image_path):
#     image = preprocess_image(image_path)

#     img = image_processor(image, return_tensors="pt").to(device)

#     output = model.generate(**img)

#     caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

#     return caption
