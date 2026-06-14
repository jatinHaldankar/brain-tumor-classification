import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os


CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Descriptions shown in the UI for each prediction
CLASS_INFO = {
    "Glioma": {
        "description": "A type of tumor that occurs in the brain and spinal cord. Gliomas begin in the glial cells that surround and support nerve cells.",
        "severity": "High",
        "color": "#ef4444"
    },
    "Meningioma": {
        "description": "A tumor that arises from the meninges — the membranes that surround the brain and spinal cord. Most meningiomas are noncancerous.",
        "severity": "Medium",
        "color": "#f97316"
    },
    "No Tumor": {
        "description": "No tumor detected in the MRI scan. The brain tissue appears normal.",
        "severity": "None",
        "color": "#22c55e"
    },
    "Pituitary": {
        "description": "A tumor that forms in the pituitary gland at the base of the brain. Most pituitary tumors are noncancerous adenomas.",
        "severity": "Medium",
        "color": "#f97316"
    },
}


class PredictionPipeline:
    """
    Loads the trained VGG16 model and runs inference on a single MRI image.
    Returns the predicted class name, confidence score, and class-level info.
    """

    def __init__(self, model_path: str = "artifacts/training/model.h5"):
        self.model_path = model_path
        self.model = None

    def _load_model(self):
        """Lazy-loads the model once and caches it on the instance."""
        if self.model is None:
            self.model = load_model(self.model_path)

    def _preprocess(self, img_path: str) -> np.ndarray:
        """
        Loads an image from disk, resizes to 224x224 (VGG16 input size),
        scales pixel values to [0, 1], and adds a batch dimension.
        """
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)   # shape: (224, 224, 3)
        img_array = img_array / 255.0          # rescale to match training
        img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)
        return img_array

    def predict(self, img_path: str) -> dict:
        """
        Runs inference on a single image and returns:
          - prediction: class name string
          - confidence: float 0–100
          - all_probs: dict of {class_name: confidence%} for all 4 classes
          - info: description and severity for the predicted class
        """
        self._load_model()
        img_array = self._preprocess(img_path)

        probs = self.model.predict(img_array)[0]   # shape: (4,)
        predicted_idx = int(np.argmax(probs))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(probs[predicted_idx]) * 100

        all_probs = {
            CLASS_NAMES[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        return {
            "prediction": predicted_class,
            "confidence": round(confidence, 2),
            "all_probs": all_probs,
            "info": CLASS_INFO[predicted_class],
        }
