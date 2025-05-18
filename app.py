from flask import Flask, request, jsonify
import cv2
import base64
import numpy as np
from flask_cors import CORS
from tf_keras.models import load_model
from openai import OpenAI

client = OpenAI(api_key="sk-35285e387ed149588260f97d7313ee3a", base_url="https://api.deepseek.com")

app = Flask(__name__)
CORS(app)

# 表情识别相关配置
EMOTION_MODEL_PATH = 'emotion_model.h5'
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# 加载人脸检测器和表情模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = load_model(EMOTION_MODEL_PATH)

@app.route('/detect_face', methods=['POST'])
def detect_face():
    data = request.get_json()
    try:
        image_base64 = data.get('image')
        # 解码Base64图像
        image_bytes = base64.b64decode(image_base64.split(',')[1])
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            # 取第一个检测到的人脸
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]  # 提取人脸区域
            
            # 预处理：调整大小、归一化
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi = np.expand_dims(face_roi, axis=-1)

            # 预测表情
            predictions = emotion_model.predict(face_roi)
            predicted_emotion = emotion_labels[np.argmax(predictions)]
            result = f"Detected emotion: {predicted_emotion}"
        else:
            result = "No face detected"

        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message')
    emotion = data.get('emotion', '')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400

    try:
        # 根据表情结果动态生成系统提示
        system_prompt = "你是一个专业的心理健康助手，需要结合用户的当前表情和消息内容提供情感支持。"
        if emotion:
            system_prompt += f" 用户当前的表情检测结果为：{emotion}，请根据此信息调整回复的语气和内容。"

        # 调用大模型时注入表情上下文
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            stream=False
        )
        return jsonify({'response': response.choices[0].message.content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)  # 关闭调试模式