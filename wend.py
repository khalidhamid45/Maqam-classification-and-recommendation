import streamlit as st
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
import tensorflow.keras.models as load
import numpy as np
import librosa
import pickle
import soundfile as sf
import pandas as pd
from fuzzywuzzy import fuzz
import random
import matplotlib.pyplot as plt  # Added for sound wave visualization

# Spotify API credentials
CLIENT_ID = "YOUR_SPOTIFY_CLIENT_ID"
CLIENT_SECRET = "YOUR_SPOTIFY_CLIENT_SECRET"

# Function to authenticate Spotify
def authenticate_spotify():
    client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    return Spotify(client_credentials_manager=client_credentials_manager)

# Define function to extract audio features
def extract_feature(file_path, mfcc=True):
    signal, sample_rate = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=13,
        n_fft=2048,
        hop_length=512
    )
    mfccs = mfccs.T
    return mfccs

# Define function to predict Maqam
def predict_maqam(file_path):
    filename = "modelForPrediction1.sav"
    loaded_model = pickle.load(open(filename, 'rb'))

    expected_length = 431
    feature = extract_feature(file_path)

    if feature.shape[0] < expected_length:
        pad_width = expected_length - feature.shape[0]
        feature = np.pad(feature, ((0, pad_width), (0, 0)), mode='constant')
    elif feature.shape[0] > expected_length:
        feature = feature[:expected_length, :]

    feature = np.expand_dims(feature, axis=0)
    feature = np.expand_dims(feature, axis=3)

    prediction = loaded_model.predict(feature)

    maqam_names = ["Ajam", "Bayat", "Hijaz", "Kurd", "Nahawand", "Rast", "Saba", "Seka"]
    predicted_class_index = np.argmax(prediction)
    predicted_maqam_name = maqam_names[predicted_class_index]

    return predicted_maqam_name

# Define function to compute similarity
def compute_similarity(reference_maqam, maqam_list):
    return [fuzz.ratio(reference_maqam.lower(), maqam.lower()) for maqam in maqam_list]

# Define function to recommend top 3 similar Maqams
def recommend_top3_similar_maqams(similarities, maqam_list, link_list):
    sorted_indices = np.argsort(similarities)[::-1]
    top3_indices = sorted_indices[:3]

    st.write("Top 3 Recommended Maqams:")
    for index in top3_indices:
        st.write(f"Maqam: {maqam_list[index]}, Listen: {link_list[index]}")

# Define function to split and predict Maqam for uploaded audio file
def split_and_predict_maqam(file_path):
    data, sample_rate = sf.read(file_path)
    total_samples = len(data)
    split_duration = 30
    split_samples = int(split_duration * sample_rate)
    num_splits = total_samples // split_samples

    predicted_maqams = []

    for i in range(num_splits):
        start = i * split_samples
        end = (i + 1) * split_samples

        split_data = data[start:end]
        split_file_path = f"split_{i}.wav"
        sf.write(split_file_path, split_data, sample_rate)

        predicted_maqam = predict_maqam(split_file_path)
        predicted_maqams.append(predicted_maqam)

    predicted_maqams_str = "-".join(predicted_maqams)
    st.write("Predicted Maqams: ", predicted_maqams_str)

    recommendations_df = pd.read_excel("maqam_recommend.xlsx")
    predicted_maqam = predict_maqam(file_path)
    filtered_recommendations = recommendations_df[recommendations_df["Maqams"].str.contains(predicted_maqam)]

    if not filtered_recommendations.empty:
        maqam_list = filtered_recommendations["Maqams"].tolist()
        link_list = filtered_recommendations["Link"].tolist()
        similarities = compute_similarity(predicted_maqam, maqam_list)
        recommend_top3_similar_maqams(similarities, maqam_list, link_list)

    st.write("Splitting and prediction complete.")

# Define function to display random recommendations
def display_random_recommendations():
    recommendations_df = pd.read_excel("maqam_recommend.xlsx")
    num_rows = recommendations_df.shape[0]
    random_indices = random.sample(range(num_rows), 2)

    st.write("Random:")
    for index in random_indices:
        maqam = recommendations_df.loc[index, "Maqams"]
        link = recommendations_df.loc[index, "Link"]
        st.write(f"Maqam: {maqam}, Link: {link}")

def main():
    st.title("Spotify Sheikh Recitations and Maqam Recommendations")
    st.write("Enter a sheikh name:")

    sheikh_name = st.text_input("Sheikh Name")

    if sheikh_name:
        spotify = authenticate_spotify()
        results = spotify.search(q=sheikh_name, type='artist')

        if results and 'artists' in results and 'items' in results['artists']:
            sheikhs = results['artists']['items']

            if len(sheikhs) > 0:
                st.write(f"Found {len(sheikhs)} sheikhs. Select a sheikh:")

                sheikh_selection = st.selectbox("Select a Sheikh", [sheikh['name'] for sheikh in sheikhs])

                selected_sheikh = None
                for sheikh in sheikhs:
                    if sheikh['name'] == sheikh_selection:
                        selected_sheikh = sheikh
                        break

                if selected_sheikh:
                    st.write("Sheikh Name:", selected_sheikh['name'])
                    st.write("Followers:", selected_sheikh['followers']['total'])
                    st.write("Popularity:", selected_sheikh['popularity'])

                    top_tracks = spotify.artist_top_tracks(selected_sheikh['id'])

                    if top_tracks and 'tracks' in top_tracks:
                        st.write("Top Recitations:")
                        for idx, track in enumerate(top_tracks['tracks']):
                            st.write(f"{idx + 1}. {track['name']} from the album {track['album']['name']}")
                            track_url = f"https://open.spotify.com/track/{track['id']}"
                            st.markdown(f"[Download {track['name']}](https://p.scdn.co/mp3-preview/{track['preview_url'].split('/')[-1]})")

                            if idx >= 5:
                                break

        else:
            st.write("No sheikhs found for the given name.")

    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

    if uploaded_file:
        st.write("Audio file uploaded!")
        predicted_maqam = predict_maqam(uploaded_file)
        st.write("Predicted Maqam:", predicted_maqam)

        st.write("---")

        st.title("Maqam Prediction and Recommendations")
        st.write("Upload an audio file and get the predicted Maqam.")

        if st.button("Predict Maqam"):
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())

            split_and_predict_maqam(uploaded_file.name)
    
    st.write("Random")
    display_random_recommendations()


if __name__ == "__main__":
    main()
