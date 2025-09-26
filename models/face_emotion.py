from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch
from PIL import Image

class FaceEmotionModel:
    def __init__(self):
        self.model_name = "ElenaRyumina/face_emotion_recognition"
        self.extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_name)

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.extractor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        label = self.model.config.id2label[probs.argmax().item()]
        return label
