import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import pickle
import os

# Set page config
st.set_page_config(page_title="Visual Question Answering", layout="wide")

# Load necessary components (cache to avoid reloading)
@st.cache_resource
def load_vqa_model():
    try:
        model = load_model('model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model.h5: {str(e)}")
        return None

@st.cache_resource
def load_tokenizer():
    try:
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer
    except:
        st.error("Could not load tokenizer.pkl")
        return None

@st.cache_data
def load_answer_classes():
    try:
        with open('answer_classes.pkl', 'rb') as f:
            answer_classes = pickle.load(f)
        return answer_classes
    except:
        st.error("Could not load answer_classes.pkl")
        return None

# Preprocess image function
def preprocess_image(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

# Preprocess question function
def preprocess_question(question, tokenizer, max_length=30):
    sequence = tokenizer.texts_to_sequences([question])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_length, padding='post')
    return padded_sequence[0]

# Main app function
def main():
    st.title("Visual Question Answering System")
    st.write("Upload an image and ask a question about it")
    
    # Load components
    model = load_vqa_model()
    tokenizer = load_tokenizer()
    answer_classes = load_answer_classes()
    
    if model is None or tokenizer is None or answer_classes is None:
        st.warning("Please ensure you have the following files in your directory:")
        st.markdown("- `model.h5` (pre-trained VQA model)")
        st.markdown("- `tokenizer.pkl` (question tokenizer)")
        st.markdown("- `answer_classes.pkl` (answer vocabulary)")
        return
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Image upload
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Question input
            question = st.text_input("Ask a question about the image:", 
                                   placeholder="e.g. What color is the car?")
            
            if st.button("Get Answer") and question:
                with st.spinner("Processing your question..."):
                    try:
                        # Preprocess image
                        processed_image = preprocess_image(image)
                        
                        # Preprocess question
                        processed_question = preprocess_question(question, tokenizer)
                        processed_question = np.expand_dims(processed_question, axis=0)
                        
                        # Make prediction
                        prediction = model.predict([processed_image, processed_question])
                        predicted_idx = np.argmax(prediction)
                        answer = answer_classes[predicted_idx]
                        confidence = prediction[0][predicted_idx] * 100
                        
                        # Display result
                        st.success(f"Answer: **{answer}** (Confidence: {confidence:.2f}%)")
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
    
    with col2:
        if uploaded_file is None:
            st.info("Sample questions you can try after uploading an image:")
            st.markdown("- What color is the object?")
            st.markdown("- Is there a person in the image?")
            st.markdown("- What is the main object in this picture?")
            st.markdown("- How many people are in the image?")
            st.markdown("- What type of animal is this?")
        else:
            st.info("VQA Model Information")
            st.markdown(f"**Model:** {model.name}")
            st.markdown(f"**Input shape:** {model.input_shape}")
            st.markdown(f"**Output shape:** {model.output_shape}")
            st.markdown(f"**Answer classes:** {len(answer_classes)} possible answers")

if __name__ == "__main__":
    main()