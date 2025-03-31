import numpy as np
import tflite_runtime.interpreter as tflite
import cv2

class MobileNetClassifier:
    def __init__(self, model_path='cv/model/mobilenet_classifier/MobileNet.tflite', 
                class_names_path='cv/model/mobilenet_classifier/class_names.txt'):
        # Load TFLite model
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Load class names
        try:
            with open(class_names_path, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f"Class names file not found: {class_names_path}")
            self.class_names = ["Unknown"]
    
    def __call__(self, image):
        """
        Process an image and return classification results.
        
        Args:
            image: RGB or BGR image in numpy array format
        
        Returns:
            predicted_class: String name of the predicted class
            confidence: Confidence score for the prediction
        """
        # Make sure image is RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            if image.dtype == np.uint8:  # Likely BGR
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        input_shape = self.input_details[0]['shape'][1:3]  # [height, width]
        processed_image = cv2.resize(image, (input_shape[1], input_shape[0]))
        
        # Normalize
        processed_image = processed_image.astype(np.float32) / 255.0
        
        # Add batch dimension
        processed_image = np.expand_dims(processed_image, axis=0)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], processed_image)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output tensor
        predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Get prediction
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get class name
        if predicted_class_idx < len(self.class_names):
            predicted_class = self.class_names[predicted_class_idx]
        else:
            predicted_class = f"Class {predicted_class_idx}"
        
        return predicted_class, confidence