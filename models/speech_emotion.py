import onnxruntime
import numpy as np
import torchaudio

class SpeechEmotionModel:
    def __init__(self, model_path="Speech-Emotion-Classification-ONNX.onnx"):
        self.session = onnxruntime.InferenceSession(model_path)

    def predict(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0).numpy()  # mono
        inputs = waveform.astype(np.float32)
        outputs = self.session.run(None, {"input": inputs})
        predicted_index = np.argmax(outputs[0])
        return predicted_index
