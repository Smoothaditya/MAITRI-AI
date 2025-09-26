from models.face_emotion import FaceEmotionModel
from models.speech_emotion import SpeechEmotionModel, map_speech_emotion
import openai

# OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Initialize models
face_model = FaceEmotionModel()
speech_model = SpeechEmotionModel()

# Map emotions to numeric trouble scores (adjust weights if needed)
emotion_scores = {
    "happy": 0,
    "calm": 1,
    "neutral": 2,
    "anxious": 3,
    "stressed": 4,
    "fatigued": 5,
    "angry": 4,
    "sad": 3,
    "fear": 4,
    "disgust": 4,
    "surprise": 1
}

def compute_trouble_score(face_emotion, speech_emotion):
    face_score = emotion_scores.get(face_emotion.lower(), 2)
    speech_score = emotion_scores.get(speech_emotion.lower(), 2)
    combined = (face_score + speech_score) / 2
    return combined

def get_conversation(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return response['choices'][0]['message']['content']

def main():
    # Example crew members and their test files
    crew_data = [
        {"name": "Crew1", "face_img": "crew1_face.jpg", "speech_audio": "crew1_audio.wav"},
        {"name": "Crew2", "face_img": "crew2_face.jpg", "speech_audio": "crew2_audio.wav"},
        {"name": "Crew3", "face_img": "crew3_face.jpg", "speech_audio": "crew3_audio.wav"}
    ]

    combined_scores = []
    status_summary = []

    for member in crew_data:
        face_label = face_model.predict(member["face_img"])
        speech_label = map_speech_emotion(speech_model.predict(member["speech_audio"]))
        trouble_score = compute_trouble_score(face_label, speech_label)
        
        combined_scores.append(trouble_score)
        status_summary.append(f"{member['name']}: face={face_label}, speech={speech_label}, score={trouble_score}")

    overall_trouble_score = sum(combined_scores) / len(combined_scores)
    summary_text = "\n".join(status_summary)
    summary_text += f"\n\nOverall Crew Trouble Score: {overall_trouble_score:.2f}"

    print(summary_text)
    assistant_response = get_conversation(summary_text)
    print("\nAssistant Response:", assistant_response)

if __name__ == "__main__":
    main()
