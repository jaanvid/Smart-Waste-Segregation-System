import pickle
import streamlit as st
import cv2
import numpy as np
import librosa
from moviepy.editor import VideoFileClip
from streamlit_lottie import st_lottie
from keras.applications.vgg16 import VGG16

st.set_page_config(layout="centered")

def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

def extract_audio_features(audio_data, mfcc_max_pad_len=100):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=48000)
    if mfccs.shape[1] < mfcc_max_pad_len:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, mfcc_max_pad_len - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :mfcc_max_pad_len]
    return mfccs.flatten()

def extract_video_features(video_path, target_size=(128, 128)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, target_size)
        frames.append(frame_rgb)
    cap.release()
    frames = np.array(frames)
    model = VGG16(weights='imagenet', include_top=False)
    features = model.predict(frames)
    return features.reshape(features.shape[0], -1)

def extract_audio_from_video(video_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_data = audio_clip.to_soundarray()[:, 0]
    video_clip.close()
    return audio_data

def calculate_brightness(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray_frame)

def main():
    # Load a Logistic Regression model
    audio_logistic_model_filename = 'Pickle Files/a_lr_model.pkl'
    audio_logistic_model = load_model(audio_logistic_model_filename)

    # Load a SVM model
    audio_svm_model_filename = 'Pickle Files/a_svm_model.pkl'
    audio_svm_model = load_model(audio_svm_model_filename)

    # Load a Random Forest model
    audio_rf_model_filename = 'Pickle Files/a_rf_model.pkl'
    audio_rf_model = load_model(audio_rf_model_filename)

    # Load a Logistic Regression model
    video_logistic_model_filename = 'Pickle Files/v_lr_model.pkl'
    video_logistic_model = load_model(video_logistic_model_filename)

    # Load a KNN model
    video_knn_model_filename = 'Pickle Files/v_knn_model.pkl'
    video_knn_model = load_model(video_knn_model_filename)

    # Load a SVM model
    video_svm_model_filename = 'Pickle Files/v_svm_model.pkl'
    video_svm_model = load_model(video_svm_model_filename)

    # Load a Random Forest model
    video_rf_model_filename = 'Pickle Files/v_rf_model.pkl'
    video_rf_model = load_model(video_rf_model_filename)

    st.title('BinGenius: A Multimodal Waste Segregation System')
    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        video_path = uploaded_file.name
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        
        st.video(video_path)

        if st.button('Predict'):
            # Load models 
            audio_models = [audio_logistic_model, audio_svm_model, audio_rf_model] 
            video_models = [video_logistic_model, video_knn_model, video_svm_model, video_rf_model]  
            
            # Predict with models
            audio_data = extract_audio_from_video(video_path)
            audio_features = extract_audio_features(audio_data)
            video_features = extract_video_features(video_path)

            cap = cv2.VideoCapture(video_path)

            # Check video brightness
            brightness_threshold = 20.0  # 50% brightness threshold
            brightness_values = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                brightness = calculate_brightness(frame)
                brightness_values.append(brightness)
            cap.release()

            average_brightness = np.mean(brightness_values)
            average_brightness = (average_brightness/255.0)*100  # value converted to %
            #st.write(f"Average brightness: {average_brightness:.2f} %")

            if average_brightness < brightness_threshold:
                #st.write("Average brightness is less than 50%. Therefore, only audio models are used for prediction.")
                audio_predictions = []
                for model in audio_models:
                    outputs = model.predict([audio_features])[0]
                    audio_predictions.append(outputs)
                audio_predictions_array = np.array(audio_predictions)
                audio_mode = np.unique(audio_predictions_array)[np.argmax(np.unique(audio_predictions_array, return_counts=True)[1])]
                #st.write("Mode of predictions from all audio models:", audio_mode)
                st.subheader(f"Prediction: {str(audio_mode)}")    
                if audio_mode == 'Recyclable':
                    st_lottie("D:/aiml/projects/s4/wasteSegregation/re.json", height=300, key="coding")
                else:
                    animation_url = "D:/aiml/projects/s4/wasteSegregation/lvgsoq6a.lottie"
                    st_lottie(animation_url, loop=True, height=300, width=400)
            else:
                #st.write("Average brightness is greater than 50%. Therefore, both audio and video models are used for prediction.")
                predictions_audio = []
                predictions_video = []
                for model in audio_models:
                    outputs = model.predict([audio_features])[0]
                    predictions_audio.append(outputs)
                    #st.write(f"Predictions from {model.__class__.__name__} (audio): {outputs}")
                for model in video_models:
                    outputs = model.predict(video_features)
                    predictions_video.append(outputs)
                    #st.write(f"Predictions from {model.__class__.__name__} (video): {outputs}")
                # Convert predictions to NumPy arrays
                predictions_audio_array = np.array(predictions_audio)
                predictions_video = np.array(predictions_video)
                predictions_video_array = predictions_video.flatten()
                predictions_array = np.concatenate((predictions_audio_array, predictions_video_array))
                # Calculate mode of predictions from video models
                mode_prediction = np.unique(predictions_array)[np.argmax(np.unique(predictions_array, return_counts=True)[1])]
                #st.write("Mode of predictions from all video and audio models:", mode_prediction)
                st.subheader(f"Prediction: {str(mode_prediction)}")
                if mode_prediction == 'Recyclable':
                    st_lottie("D:/aiml/projects/s4/wasteSegregation/re.json", height=300, key="coding")
                else:
                    with open("D:/aiml/projects/s4/wasteSegregation/lvgsoq6a.lottie", "r", encoding="utf-8") as f:
                        animation_data = f.read()
                    st_lottie(animation_data, loop=True, autoplay=True, height=300, width=400)

if __name__ == "__main__":
    main()
