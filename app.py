import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from easy_vqa import get_test_questions, get_test_image_paths
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    return keras.models.load_model('model.h5')

# Load tokenizer and answers list
@st.cache_data
def load_tokenizer_and_answers():
    from easy_vqa import get_answers
    answers_list = get_answers()
    
    # Create tokenizer (same as in notebook)
    tokenizer = Tokenizer(num_words=35, oov_token='<OOV>')
    train_questions, _, _ = get_test_questions()  # Using test as we don't have train in this context
    tokenizer.fit_on_texts(train_questions)
    
    return tokenizer, answers_list

# Load and preprocess image
def preprocess_image(image_path):
    img = img_to_array(load_img(image_path))
    return img / 255.0

# Main app
def main():
    st.title("Easy-VQA Visual Question Answering")
    st.write("This app answers questions about simple images containing shapes and colors.")
    
    # Load model and resources
    model = load_model()
    tokenizer, answers_list = load_tokenizer_and_answers()
    
    # Get test data
    test_questions, test_answers, test_image_ids = get_test_questions()
    test_image_paths = get_test_image_paths()
    
    # Create a dictionary of test images
    test_imgs = {image_id: preprocess_image(image_path) 
                for image_id, image_path in test_image_paths.items()}
    
    # Sidebar controls
    st.sidebar.header("Controls")
    sample_idx = st.sidebar.slider("Select a test sample", 0, len(test_questions)-1, 0)
    
    # Display selected sample
    image_id = test_image_ids[sample_idx]
    question = test_questions[sample_idx]
    true_answer = test_answers[sample_idx]
    
    st.subheader("Image")
    img_array = test_imgs[image_id]
    st.image(img_array, width=300)
    
    st.subheader("Question")
    st.write(question)
    
    st.subheader("True Answer")
    st.write(true_answer)
    
    # Prepare input for model
    seq = tokenizer.texts_to_sequences([question])
    padded_seq = pad_sequences(seq, padding='post', maxlen=9)
    img_input = np.expand_dims(test_imgs[image_id], axis=0)
    
    # Make prediction
    if st.button("Predict Answer"):
        pred = model.predict([img_input, padded_seq])
        pred_idx = np.argmax(pred)
        pred_answer = answers_list[pred_idx]
        
        st.subheader("Predicted Answer")
        st.write(pred_answer)
        
        # Show confidence
        confidence = pred[0][pred_idx]
        st.write(f"Confidence: {confidence:.2%}")
        
        # Show top 3 predictions
        st.subheader("Top Predictions")
        top_k = 3
        top_indices = np.argsort(pred[0])[-top_k:][::-1]
        
        for i, idx in enumerate(top_indices):
            st.write(f"{i+1}. {answers_list[idx]} ({pred[0][idx]:.2%})")

if __name__ == "__main__":
    main()