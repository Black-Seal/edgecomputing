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
import gc

from speechtospeech import Can_to_eng, Eng_to_can, MultilingualTranslator
from cv.utils.cvfpscalc import CvFpsCalc
from cv.model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from cv.model.point_history_classifier.point_history_classifier import PointHistoryClassifier

# Lazy loading for MobileNet with caching
@st.cache_resource
def import_mobilenet():
    try:
        from cv.model.mobilenet_classifier.mobilenet_classifier import MobileNetClassifier
        return MobileNetClassifier
    except Exception as e:
        st.error(f"Failed to import MobileNet: {str(e)}")
        return None

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
    
    # Hand Gesture tab - Fully optimized for Raspberry Pi
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
        
        # Option to enable MobileNet
        use_mobilenet = st.checkbox("Enable Sign Language Recognition (MobileNet)", value=False, key="use_mobilenet")
        
        # If hand gesture recognition is active, show the webcam feed
        if st.session_state.hand_gesture_active:
            # Initialize MediaPipe Hands with optimized settings for Pi
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,  # Limit to 1 hand for better performance
                min_detection_confidence=0.5,  # Lower threshold for better performance
                min_tracking_confidence=0.5,
            )
            
            # Initialize the classifiers
            keypoint_classifier = KeyPointClassifier()
            point_history_classifier = PointHistoryClassifier()
            
            # Conditionally initialize MobileNet
            mobilenet_active = False
            mobilenet_classifier = None
            
            if use_mobilenet:
                with st.spinner("Loading MobileNet for sign language recognition..."):
                    try:
                        MobileNetClassifier = import_mobilenet()
                        if MobileNetClassifier is not None:
                            mobilenet_classifier = MobileNetClassifier()
                            st.success("MobileNet loaded successfully!")
                            mobilenet_active = True
                    except Exception as e:
                        st.error(f"Failed to load MobileNet: {str(e)}")
                        mobilenet_active = False
            
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
            if mobilenet_active:
                gesture_info_cols = st.columns(3)
                with gesture_info_cols[2]:
                    sign_language_placeholder = st.empty()
            else:
                gesture_info_cols = st.columns(2)
                
            with gesture_info_cols[0]:
                hand_sign_placeholder = st.empty()
            with gesture_info_cols[1]:
                finger_gesture_placeholder = st.empty()
            
            # Create a placeholder for the video frame
            frame_placeholder = st.empty()
            
            # Use OpenCV to capture video with reduced resolution for Pi
            cap = cv.VideoCapture(0)
            cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)  # Lower resolution for Pi
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
            
            # FPS display
            fps_text = st.empty()
            
            try:
                # Initialize frame counters for skipping
                frame_count = 0
                sign_frame_counter = 0
                
                while cap.isOpened() and st.session_state.hand_gesture_active:
                    fps = cvFpsCalc.get()
                    fps_text.text(f"FPS: {fps:.1f}")
                    
                    # Skip every other frame for better performance
                    frame_count += 1
                    if frame_count % 2 != 0:
                        ret, _ = cap.read()  # Read but don't process
                        continue
                    
                    ret, image = cap.read()
                    if not ret:
                        st.error("Failed to capture video. Please check your camera.")
                        break
                        
                    image = cv.flip(image, 1)  # Mirror display
                    
                    # Use shallow copy for better performance
                    debug_image = image.copy()
                    
                    # Process with MediaPipe
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
                            
                            # Simplified drawing for Raspberry Pi
                            debug_image = draw_landmarks_simplified(debug_image, landmark_list)
                            if brect[2] > brect[0] and brect[3] > brect[1]:
                                cv.rectangle(debug_image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 1)
                            
                            most_common_gesture_id = most_common_fg_id[0][0] if most_common_fg_id else 0
                            
                            hand_sign_placeholder.markdown(f"**Hand Sign:** {keypoint_classifier_labels[hand_sign_id]}")
                            finger_gesture_placeholder.markdown(f"**Finger Gesture:** {point_history_classifier_labels[most_common_gesture_id]}")
                            
                            # MobileNet classification for sign language (process less frequently)
                            sign_frame_counter += 1
                            if mobilenet_active and mobilenet_classifier is not None and sign_frame_counter % 5 == 0:
                                try:
                                    # Extract hand region with margin
                                    margin = 30
                                    x1 = max(0, brect[0] - margin)
                                    y1 = max(0, brect[1] - margin)
                                    x2 = min(debug_image.shape[1], brect[2] + margin)
                                    y2 = min(debug_image.shape[0], brect[3] + margin)
                                    
                                    # Only process if we have a valid hand image
                                    if x2 > x1 and y2 > y1:
                                        hand_img = debug_image[y1:y2, x1:x2]
                                        
                                        if hand_img.size > 0:
                                            # Get sign language classification
                                            sign, confidence = mobilenet_classifier(hand_img)
                                            
                                            # Display result
                                            sign_language_placeholder.markdown(f"**Sign Language:** {sign} ({confidence:.2f})")
                                            
                                            # Simplified text display
                                            text_position = (brect[0], brect[3] + 20)
                                            cv.putText(debug_image, 
                                                    f"Sign: {sign}", 
                                                    text_position, 
                                                    cv.FONT_HERSHEY_SIMPLEX, 
                                                    0.5, (0, 255, 0), 1, cv.LINE_AA)
                                except Exception as e:
                                    print(f"Error in MobileNet classification: {str(e)}")
                    else:
                        point_history.append([0, 0])
                        if mobilenet_active:
                            sign_language_placeholder.markdown("**Sign Language:** None")
                    
                    # Convert to RGB for display
                    debug_image_rgb = cv.cvtColor(debug_image, cv.COLOR_BGR2RGB)
                    
                    # Display the frame
                    frame_placeholder.image(debug_image_rgb, channels="RGB", use_container_width=True)
                    
                    # Longer sleep for Pi to prevent CPU overload
                    time.sleep(0.03)
                    
                    # Periodic garbage collection
                    if frame_count % 100 == 0:
                        gc.collect()
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
            finally:
                if 'cap' in locals() and cap is not None:
                    cap.release()
                st.session_state.hand_gesture_active = False
                # Force garbage collection
                gc.collect()
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
            
            st.success("Optimized for Raspberry Pi performance.")
    
    # History tab
    with tab3:
        st.subheader(f"Translation History ({st.session_state.translation_count})")
        if not st.session_state.history:
            st.info("No translations yet. Start translating to build your history!")
        else:
            # Limit history display for better performance
            history_to_show = st.session_state.history[-5:]
            
            for i, item in enumerate(reversed(history_to_show)):
                with st.expander(f"{item['timestamp']} - {item['direction']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Input:**")
                        st.info(item['input'])
                    with col2:
                        st.markdown("**Output:**")
                        st.success(item['output'])
            
            if len(st.session_state.history) > 5:
                st.info(f"Showing only the 5 most recent translations. {len(st.session_state.history) - 5} older entries are hidden for better performance.")

# Simplified drawing function optimized for Raspberry Pi
def draw_landmarks_simplified(image, landmark_point):
    if len(landmark_point) > 0:
        # Draw only key points (base of palm and fingertips)
        key_points = [0, 4, 8, 12, 16, 20]
        for index in key_points:
            if index < len(landmark_point):
                cv.circle(image, (landmark_point[index][0], landmark_point[index][1]), 
                          5, (0, 255, 0), -1)
    return image

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

def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), 2)
    return image

if __name__ == "__main__":
    main()