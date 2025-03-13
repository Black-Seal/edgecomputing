import streamlit as st
import pandas as pd
import numpy as np
import time
import cv2 as cv
import mediapipe as mp
import copy
import itertools
from collections import Counter, deque
import csv

from speechtospeech import Can_to_eng, Eng_to_can, MultilingualTranslator
from cv.utils.cvfpscalc import CvFpsCalc
from cv.model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from cv.model.point_history_classifier.point_history_classifier import PointHistoryClassifier

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Speech Translator",
        page_icon="🗣️",
        layout="centered"
    )
    
    # App title with styling
    st.markdown("""
    # 🗣️ Speech Translator
    Translate between Cantonese-English or any supported languages using your voice
    """)
    
    # Create session state variables
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""
    if 'output_text' not in st.session_state:
        st.session_state.output_text = ""
    if 'translation_count' not in st.session_state:
        st.session_state.translation_count = 0
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Translator", "Hand Gesture", "History"])
    
    with tab1:
        st.subheader("Select Translation Mode")
        mode = st.radio(
            "Choose a mode:",
            ("Cantonese-English Translation", "International Translation")
        )
        
        if mode == "Cantonese-English Translation":
            st.subheader("Select Translation Direction")
            choice = st.radio(
                "Choose an option:",
                ("Speak in Cantonese → Translate to English",
                 "Speak in English → Translate to Cantonese")
            )
            
            # Translation section
            col1, col2 = st.columns([2, 1])
            with col1:
                translate_button = st.button("Start Recording & Translate", type="primary", key="can_eng")
            with col2:
                clear_button = st.button("Clear Results", key="clear_can_eng")
                if clear_button:
                    st.session_state.input_text = ""
                    st.session_state.output_text = ""
            
            if translate_button:
                with st.status("Processing...", expanded=True) as status:
                    st.write("🎤 Recording audio...")
                    
                    if choice == "Speak in Cantonese → Translate to English":
                        cantonese_text, english_translation = Can_to_eng.translate_cantonese_to_english_speech_with_return()
                        st.session_state.input_text = cantonese_text if cantonese_text else "No speech detected"
                        st.session_state.output_text = english_translation if english_translation else "Translation failed"
                    else:
                        english_text, cantonese_translation = Eng_to_can.translate_english_to_cantonese_speech_with_return()
                        st.session_state.input_text = english_text if english_text else "No speech detected"
                        st.session_state.output_text = cantonese_translation if cantonese_translation else "Translation failed"
                    
                    st.write("✅ Translation complete!")
                    status.update(label="Translation complete!", state="complete")
                    
                    # Add to history
                    if st.session_state.input_text and st.session_state.output_text:
                        timestamp = time.strftime("%H:%M:%S")
                        st.session_state.history.append({
                            "timestamp": timestamp,
                            "direction": choice,
                            "input": st.session_state.input_text,
                            "output": st.session_state.output_text
                        })
                        st.session_state.translation_count += 1
            
            # Display results
            st.subheader("Translation Results")
            results_cols = st.columns(2)
            with results_cols[0]:
                source_lang = "Cantonese" if choice == "Speak in Cantonese → Translate to English" else "English"
                st.markdown(f"##### Source: {source_lang}")
                st.info(st.session_state.input_text if st.session_state.input_text else "Input will appear here")
            with results_cols[1]:
                target_lang = "English" if choice == "Speak in Cantonese → Translate to English" else "Cantonese"
                st.markdown(f"##### Target: {target_lang}")
                st.success(st.session_state.output_text if st.session_state.output_text else "Translation will appear here")
        
        elif mode == "International Translation":
            st.subheader("Select Languages")
            languages = {
                "Arabic": "ar",
                "Bahasa (Indonesian)": "id",
                "Bengali": "bn",
                "Bulgarian": "bg",
                "Chinese (Mandarin)": "zh",
                "Croatian": "hr",
                "Czech": "cs",
                "Danish": "da",
                "Dutch": "nl",
                "English": "en",
                "Finnish": "fi",
                "French": "fr",
                "German": "de",
                "Greek": "el",
                "Hindi": "hi",
                "Hungarian": "hu",
                "Italian": "it",
                "Japanese": "ja",
                "Korean": "ko",
                "Melayu (Malay)": "ms",
                "Norwegian": "no",
                "Polish": "pl",
                "Portuguese": "pt",
                "Romanian": "ro",
                "Russian": "ru",
                "Spanish": "es",
                "Swahili": "sw",
                "Swedish": "sv",
                "Tamil": "ta",
                "Thai": "th",
                "Turkish": "tr",
                "Vietnamese": "vi"
            }
            
            col1, col2 = st.columns(2)
            with col1:
                source_lang = st.selectbox("Source Language", list(languages.keys()), key="source_intl")
            with col2:
                target_lang = st.selectbox("Target Language", list(languages.keys()), index=10, key="target_intl")
            
            # Translation section
            col1, col2 = st.columns([2, 1])
            with col1:
                translate_button = st.button("Start Recording & Translate", type="primary", key="intl")
            with col2:
                clear_button = st.button("Clear Results", key="clear_intl")
                if clear_button:
                    st.session_state.input_text = ""
                    st.session_state.output_text = ""
            
            if translate_button:
                with st.status("Processing...", expanded=True) as status:
                    st.write("🎤 Recording audio...")
                    
                    translator = MultilingualTranslator()
                    source_code = languages[source_lang]
                    target_code = languages[target_lang]
                    
                    input_text, translated_text = translator.translate_speech(
                        source_lang=source_code, 
                        target_lang=target_code
                    )
                    
                    st.session_state.input_text = input_text if input_text else "No speech detected"
                    st.session_state.output_text = translated_text if translated_text else "Translation failed"
                    
                    st.write("✅ Translation complete!")
                    status.update(label="Translation complete!", state="complete")
                    
                    # Add to history
                    if st.session_state.input_text and st.session_state.output_text:
                        timestamp = time.strftime("%H:%M:%S")
                        st.session_state.history.append({
                            "timestamp": timestamp,
                            "direction": f"{source_lang} → {target_lang}",
                            "input": st.session_state.input_text,
                            "output": st.session_state.output_text
                        })
                        st.session_state.translation_count += 1
            
            # Display results
            st.subheader("Translation Results")
            results_cols = st.columns(2)
            with results_cols[0]:
                st.markdown(f"##### Source: {source_lang}")
                st.info(st.session_state.input_text if st.session_state.input_text else "Input will appear here")
            with results_cols[1]:
                st.markdown(f"##### Target: {target_lang}")
                st.success(st.session_state.output_text if st.session_state.output_text else "Translation will appear here")
    
    # Hand Gesture tab - Fully restored
    with tab2:
        st.subheader("Hand Gesture Recognition")
        st.write("Use hand gestures to control the application")
        
        # Add session state variables for hand gesture recognition
        if 'hand_gesture_active' not in st.session_state:
            st.session_state.hand_gesture_active = False
        
        # Button to toggle hand gesture recognition
        gesture_col1, gesture_col2 = st.columns([2, 1])
        with gesture_col1:
            if not st.session_state.hand_gesture_active:
                start_gesture = st.button("Start Hand Gesture Recognition", key="start_gesture")
                if start_gesture:
                    st.session_state.hand_gesture_active = True
                    st.rerun()
            else:
                stop_gesture = st.button("Stop Hand Gesture Recognition", key="stop_gesture")
                if stop_gesture:
                    st.session_state.hand_gesture_active = False
                    st.rerun()
        
        # If hand gesture recognition is active, show the webcam feed
        if st.session_state.hand_gesture_active:
            # Initialize MediaPipe Hands
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            )
            
            # Initialize the classifiers
            keypoint_classifier = KeyPointClassifier()
            point_history_classifier = PointHistoryClassifier()
            
            # Read labels
            keypoint_classifier_labels = []
            point_history_classifier_labels = []
            
            try:
                with open("cv/model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig") as f:
                    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
                with open("cv/model/point_history_classifier/point_history_classifier_label.csv", encoding="utf-8-sig") as f:
                    point_history_classifier_labels = [row[0] for row in csv.reader(f)]
                st.write("Successfully loaded label files")
            except FileNotFoundError as e:
                st.error(f"Label files not found: {str(e)}")
                keypoint_classifier_labels = ["Open", "Close", "Point", "OK"]
                point_history_classifier_labels = ["None", "Clockwise", "Counter Clockwise"]
            
            # Initialize variables
            cvFpsCalc = CvFpsCalc(buffer_len=10)
            history_length = 16
            point_history = deque(maxlen=history_length)
            finger_gesture_history = deque(maxlen=history_length)
            
            # Display gesture information
            st.markdown("### Current Gestures")
            gesture_info_col1, gesture_info_col2 = st.columns(2)
            with gesture_info_col1:
                hand_sign_placeholder = st.empty()
            with gesture_info_col2:
                finger_gesture_placeholder = st.empty()
            
            # Create a placeholder for the video frame
            frame_placeholder = st.empty()
            
            # Use OpenCV to capture video
            cap = cv.VideoCapture(0)
            cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
            
            try:
                while cap.isOpened() and st.session_state.hand_gesture_active:
                    fps = cvFpsCalc.get()
                    ret, image = cap.read()
                    if not ret:
                        st.error("Failed to capture video. Please check your camera.")
                        break
                        
                    image = cv.flip(image, 1)  # Mirror display
                    debug_image = copy.deepcopy(image)
                    
                    # Detection implementation
                    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = hands.process(image)
                    image.flags.writeable = True
                    
                    if results.multi_hand_landmarks:
                        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                            brect = calc_bounding_rect(debug_image, hand_landmarks)
                            landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                            pre_processed_landmark_list = pre_process_landmark(landmark_list)
                            pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
                            
                            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                            
                            if hand_sign_id == 2:  # Point gesture
                                point_history.append(landmark_list[8])
                            else:
                                point_history.append([0, 0])
                                
                            finger_gesture_id = 0
                            if len(pre_processed_point_history_list) == (history_length * 2):
                                finger_gesture_id = point_history_classifier(pre_processed_point_history_list)
                            
                            finger_gesture_history.append(finger_gesture_id)
                            most_common_fg_id = Counter(finger_gesture_history).most_common()
                            
                            debug_image = draw_bounding_rect(True, debug_image, brect)
                            debug_image = draw_landmarks(debug_image, landmark_list)
                            
                            most_common_gesture_id = most_common_fg_id[0][0] if most_common_fg_id else 0
                            
                            debug_image = draw_info_text(
                                debug_image,
                                brect,
                                handedness,
                                keypoint_classifier_labels[hand_sign_id],
                                point_history_classifier_labels[most_common_gesture_id]
                            )
                            
                            hand_sign_placeholder.markdown(f"**Hand Sign:** {keypoint_classifier_labels[hand_sign_id]}")
                            finger_gesture_placeholder.markdown(f"**Finger Gesture:** {point_history_classifier_labels[most_common_gesture_id]}")
                    else:
                        point_history.append([0, 0])
                        
                    debug_image = draw_point_history(debug_image, point_history)
                    debug_image = draw_info(debug_image, fps, 0, 0)
                    
                    debug_image = cv.cvtColor(debug_image, cv.COLOR_BGR2RGB)
                    time.sleep(0.01)  # Prevent CPU overload
                    frame_placeholder.image(debug_image, channels="RGB")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                if 'cap' in locals() and cap is not None:
                    cap.release()
                st.session_state.hand_gesture_active = False
        else:
            st.info("""
            Click "Start Hand Gesture Recognition" to activate the webcam and begin detecting hand gestures.
            
            Available gestures:
            - Open hand
            - Closed fist
            - Pointing
            - OK sign
            
            You can also detect circular motions with your index finger.
            """)
    
    # History tab
    with tab3:
        st.subheader(f"Translation History ({st.session_state.translation_count})")
        if not st.session_state.history:
            st.info("No translations yet. Start translating to build your history!")
        else:
            for i, item in enumerate(reversed(st.session_state.history)):
                with st.expander(f"{item['timestamp']} - {item['direction']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Input:**")
                        st.info(item['input'])
                    with col2:
                        st.markdown("**Output:**")
                        st.success(item['output'])

# Helper functions for hand gesture recognition
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    temp_point_history = copy.deepcopy(point_history)
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]
        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))
    return temp_point_history

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        connections = [
            (2, 3), (3, 4),  # Thumb
            (5, 6), (6, 7), (7, 8),  # Index finger
            (9, 10), (10, 11), (11, 12),  # Middle finger
            (13, 14), (14, 15), (15, 16),  # Ring finger
            (17, 18), (18, 19), (19, 20),  # Little finger
            (0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0)  # Palm
        ]
        for connection in connections:
            cv.line(image, tuple(landmark_point[connection[0]]), 
                    tuple(landmark_point[connection[1]]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[connection[0]]), 
                    tuple(landmark_point[connection[1]]), (255, 255, 255), 2)
        for index, landmark in enumerate(landmark_point):
            if index in [4, 8, 12, 16, 20]:  # Fingertips
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            else:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
    return image

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image

def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ": " + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image

def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), 2)
    return image

def draw_info(image, fps, mode, number):
    cv.putText(image, f"FPS: {fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, f"FPS: {fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
    return image

if __name__ == "__main__":
    main()