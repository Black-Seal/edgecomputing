import speech_recognition as sr
from deep_translator import GoogleTranslator
from gtts import gTTS
import os
import tempfile
from pydub import AudioSegment
from pydub.playback import play
import time

class SimpleTranslator:
    """
    A lightweight speech translator optimized for Raspberry Pi
    that supports translation between English and Chinese/Malay/Tamil.
    """
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # Adjust these parameters for Raspberry Pi performance
        self.recognizer.energy_threshold = 300  # Increase sensitivity
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8  # Shorter pauses between phrases
        
        # Language code mapping
        self.language_codes = {
            "en": "en-US",  # English
            "zh": "zh-CN",  # Chinese
            "ms": "ms-MY",  # Malay
            "ta": "ta-IN"   # Tamil
        }
        
        # Recognition language mapping (specifically for speech recognition)
        self.recognition_codes = {
            "en": "en-US",    # English
            "zh": "zh-CN",    # Chinese (Mandarin)
            "ms": "ms-MY",    # Malay
            "ta": "ta-IN"     # Tamil
        }
        
        # Translation language mapping (specifically for Google Translator)
        self.translation_codes = {
            "en": "en",      # English
            "zh": "zh-CN",   # Chinese
            "ms": "ms",      # Malay
            "ta": "ta"       # Tamil
        }
        
        # TTS language mapping (specifically for gTTS)
        self.tts_codes = {
            "en": "en",      # English
            "zh": "zh-CN",   # Chinese
            "ms": "ms",      # Malay
            "ta": "ta"       # Tamil
        }
        
    def record_audio(self, source_lang_code):
        """Record audio from microphone with adjusted parameters for RPi"""
        with sr.Microphone() as source:
            print(f"Listening... Speak now!")
            # Use shorter adjustment time for quicker response
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                # Shorter timeout for better responsiveness
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                return audio
            except sr.WaitTimeoutError:
                print("No speech detected within timeout period")
                return None
    
    def speech_to_text(self, audio, source_lang_code):
        """Convert speech to text using Google's API"""
        if not audio:
            return None
        
        try:
            # Use the proper recognition language code
            recognition_lang = self.recognition_codes.get(source_lang_code, "en-US")
            text = self.recognizer.recognize_google(audio, language=recognition_lang)
            print(f"Original text: {text}")
            return text
        except sr.UnknownValueError:
            print("Could not understand the audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None
    
    def translate_text(self, text, source_lang_code, target_lang_code):
        """Translate text between languages using lightweight Google API"""
        if not text:
            return None
            
        try:
            # Get the correct translation codes
            source_code = self.translation_codes.get(source_lang_code, source_lang_code)
            target_code = self.translation_codes.get(target_lang_code, target_lang_code)
            
            print(f"Translating from {source_code} to {target_code}")
            translator = GoogleTranslator(source=source_code, target=target_code)
            translated = translator.translate(text)
            print(f"Translated text: {translated}")
            return translated
        except Exception as e:
            print(f"Translation error: {e}")
            print(f"Attempting fallback method...")
            
            try:
                # Try with simplified language codes
                if source_lang_code == "zh" and target_lang_code == "en":
                    translator = GoogleTranslator(source="zh-CN", target="en")
                elif source_lang_code == "en" and target_lang_code == "zh":
                    translator = GoogleTranslator(source="en", target="zh-CN")
                else:
                    translator = GoogleTranslator(source=source_lang_code, target=target_lang_code)
                    
                translated = translator.translate(text)
                print(f"Fallback translation: {translated}")
                return translated
            except Exception as e2:
                print(f"Fallback translation error: {e2}")
                return None
    
    def text_to_speech(self, text, lang_code):
        """Convert text to speech with minimal file operations"""
        if not text:
            return False
            
        # Map the language code to the TTS format if needed
        tts_lang = self.tts_codes.get(lang_code, lang_code)
        
        try:
            print(f"Converting text to speech in language: {tts_lang}")
            # Use tempfile to avoid filesystem clutter
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_filename = temp_file.name
                
            # Create and save the audio file
            tts = gTTS(text=text, lang=tts_lang, slow=False)
            tts.save(temp_filename)
            
            # Play audio
            audio = AudioSegment.from_mp3(temp_filename)
            play(audio)
            
            # Clean up
            try:
                os.remove(temp_filename)
            except:
                pass  # Ignore cleanup errors
                
            return True
        except Exception as e:
            print(f"Text-to-speech error: {e}")
            print(f"Attempting fallback method...")
            
            # Try with alternative Chinese code if needed
            if lang_code == "zh":
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                        temp_filename = temp_file.name
                    
                    tts = gTTS(text=text, lang="zh", slow=False)
                    tts.save(temp_filename)
                    
                    audio = AudioSegment.from_mp3(temp_filename)
                    play(audio)
                    
                    try:
                        os.remove(temp_filename)
                    except:
                        pass
                    
                    return True
                except Exception as e2:
                    print(f"Fallback TTS error: {e2}")
            
            return False
    
    def translate_speech(self, source_lang="en", target_lang="zh"):
        """Complete pipeline to process speech translation optimized for RPi"""
        start_time = time.time()
        
        # Record audio
        audio = self.record_audio(source_lang)
        if not audio:
            return None, None
            
        # Convert speech to text
        original_text = self.speech_to_text(audio, source_lang)
        if not original_text:
            return None, None
            
        # Translate text
        translated_text = self.translate_text(original_text, source_lang, target_lang)
        if not translated_text:
            return original_text, None
            
        # Convert translated text to speech
        self.text_to_speech(translated_text, target_lang)
        
        elapsed_time = time.time() - start_time
        print(f"Total translation time: {elapsed_time:.2f} seconds")
        
        return original_text, translated_text


# For standalone testing
if __name__ == "__main__":
    translator = SimpleTranslator()
    
    print("Simple Speech Translator for Raspberry Pi")
    print("Supported languages: English, Chinese, Malay, Tamil")
    
    while True:
        print("\nSelect translation direction:")
        print("1. English to Chinese")
        print("2. Chinese to English")
        print("3. English to Malay")
        print("4. Malay to English")
        print("5. English to Tamil")
        print("6. Tamil to English")
        print("7. Exit")
        
        try:
            choice = input("Enter your choice (1-7): ")
            
            language_pairs = {
                "1": ("en", "zh"),
                "2": ("zh", "en"),
                "3": ("en", "ms"),
                "4": ("ms", "en"),
                "5": ("en", "ta"),
                "6": ("ta", "en"),
            }
            
            if choice == "7":
                print("Exiting...")
                break
                
            if choice in language_pairs:
                source, target = language_pairs[choice]
                print(f"\nTranslating from {source} to {target}...")
                original, translated = translator.translate_speech(source, target)
                
                if original and translated:
                    print("\nTranslation complete!")
                else:
                    print("\nTranslation failed or no speech detected.")
            else:
                print("Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")