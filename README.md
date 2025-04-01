# Speech Translator with Hand Gesture Recognition

This project provides a speech-to-speech translation application with hand gesture recognition capabilities, optimized for Raspberry Pi. The application allows users to translate between English, Chinese, Malay, and Tamil, while also incorporating hand gesture controls for a more intuitive user experience.

## Edge Computing Project

This is an edge computing project designed to run on Raspberry Pi hardware. Edge computing focuses on processing data near the source of data generation instead of relying on cloud-based servers, offering several advantages:

- **Reduced Latency**: By processing translations locally or with minimal API calls, our solution provides faster response times compared to purely cloud-based solutions
- **Improved Privacy**: Speech processing happens on the device when possible, reducing data transmission to external servers
- **Offline Capabilities**: Some functionality works with limited or no internet connectivity
- **Resource Efficiency**: Optimized to run on low-power Raspberry Pi hardware while still providing responsive performance

## Features

- **Speech-to-Speech Translation**: Translate speech between multiple languages (English, Chinese, Malay, Tamil)
- **Hand Gesture Recognition**: Control the application using hand gestures
- **Streamlit UI**: User-friendly interface with multiple tabs for different functionalities
- **Raspberry Pi Optimization**: Performance tweaks to ensure smooth operation on Raspberry Pi hardware

## Project Structure

```
EDGECOMPUTING/
├── env/                    # Virtual environment directory
└── src/                    # Source code
    ├── cv/                 # Computer vision modules
    │   ├── model/          # Hand gesture and keypoint classification models
    │   ├── utils/          # Utilities for CV operations
    │   └── hand_recog.py/  # Hand recognition implementation
    ├── speechtospeech/     # Speech translation modules
    │   ├── __pycache__/    
    │   ├── __init__.py
    │   └── simple_translator.py  # Core translation functionality
    └── app.py              # Main application file with Streamlit UI
```

## Development Branches

The project has evolved through several development branches:

- **dev-branch**: Initial development branch where work started with English to Cantonese translation and vice versa.

- **sherwyn**: Added multilingual capabilities and optimized for macOS and Windows. This branch used TensorFlow and PyTorch for more advanced models but wasn't optimized for Raspberry Pi.

- **kenan**: Explored using FastAPI instead of Streamlit for the UI, aiming for better Raspberry Pi performance. Despite faster startup times, the translation accuracy decreased, so this approach was ultimately not used.

- **main**: The final production branch focused on Singapore's main languages (English, Chinese, Malay, Tamil). To improve performance on Raspberry Pi, PyTorch and TensorFlow were replaced with TensorFlowLite for computer vision, and API calls are used for speech-to-speech translation to avoid lengthy model download times.

## Core Components

### 1. Speech Translator (`simple_translator.py`)
A lightweight speech translator that supports translation between English, Chinese, Malay, and Tamil. It uses:
- Speech recognition to capture audio input
- Google Translator API for text translation
- gTTS (Google Text-to-Speech) for audio output

### 2. Hand Gesture Recognition (`hand_recog.py`)
Implements real-time hand gesture recognition using:
- MediaPipe for hand tracking
- Custom-trained models for gesture classification
- OpenCV for image processing and visualization

### 3. Main Application (`app.py`)
The Streamlit-based user interface with three tabs:
- **Translator**: Speech translation functionality
- **Hand Gesture**: Hand gesture recognition controls
- **History**: Log of previous translations

## Performance Optimizations for Edge Computing on Raspberry Pi

The project has been carefully optimized to perform well on resource-constrained edge devices like the Raspberry Pi:

- **Lower resolution video capture** (320x240) to reduce processing requirements
- **Frame skipping** to reduce CPU load and maintain responsive UX
- **Simplified drawing functions** for hand gesture visualization
- **Proactive memory management** with garbage collection to prevent crashes
- **Single-hand tracking** instead of multi-hand to reduce computational demands
- **TensorFlowLite** for efficient computer vision tasks on edge devices
- **Strategic use of API calls** for complex translation tasks while keeping simpler processing local
- **Optimized startup time** by avoiding large model downloads
- **Balanced workload distribution** between local processing and cloud APIs

## Getting Started

1. Clone the repository
2. Create and activate a virtual environment
   ## For Raspberry Pi
   ```
   # Create the virtual environment
   python3 -m venv env

   # Activate the virtual environment
   source env/bin/activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Navigate to the src directory:
   ```
   cd src/
   ```
   > **IMPORTANT**: You must run the application from the src directory for the cv module paths to work correctly!

5. Run the application:
   ```
   streamlit run app.py
   ```

## Requirements

- Python 3.7+
- Raspberry Pi 4 or newer (recommended)
- USB Webcam with inbuilt microphone
- Bluetooth speaker or USB speaker
- Internet connection for translation API access

## Dependencies

- streamlit
- pandas
- numpy
- opencv-python
- mediapipe
- SpeechRecognition
- deep-translator
- gtts
- pydub

---

## Why Edge Computing for Speech Translation?

Edge computing is particularly well-suited for this speech translation application for several reasons:

1. **Real-time interaction requirements**: Speech translation needs to be responsive for natural conversation flow, making the low latency of edge computing essential.

2. **Privacy considerations**: Processing sensitive speech data locally when possible helps protect user privacy.

3. **Accessibility**: By running on a Raspberry Pi, the solution is affordable and can be deployed in various settings like information counters, schools, tourist locations, or community centers.

4. **Practical application**: This demonstrates how powerful NLP applications can run on affordable edge devices, making advanced translation technology more accessible.

5. **Hardware constraints as design drivers**: The Raspberry Pi's limitations led to creative optimizations and a hybrid approach (local processing + strategic API usage) that ultimately improved the user experience.

This project was developed as a solution for multilingual communication in Singapore, focusing on the primary languages spoken in the country (English, Chinese, Malay, and Tamil).
