from flask import Flask, request, jsonify, Response
import cv2
import base64
import numpy as np
from flask_cors import CORS
import requests
import re
from PIL import Image
import io
from deepface import DeepFace
import dlib
import json
import logging
from tf_keras.models import load_model

app = Flask(__name__)
CORS(app)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama API 配置
OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "deepseek-r1:7b"

# 加载dlib的人脸检测器和关键点检测器
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 表情识别相关配置
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def preprocess_image(image):
    # 1. 自适应直方图均衡化（CLAHE）
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    equalized = clahe.apply(image)
    
    # 2. 高斯模糊去噪
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    
    # 3. 对比度增强
    alpha = 1.2  # 对比度增强因子
    beta = 10    # 亮度增强因子
    enhanced = cv2.convertScaleAbs(blurred, alpha=alpha, beta=beta)
    
    return enhanced

def clean_response(text):
    # 移除所有HTML标签及其内容
    text = re.sub(r'<[^>]*>.*?</[^>]*>', '', text)
    # 移除剩余的单个HTML标签
    text = re.sub(r'<[^>]*>', '', text)
    # 移除多余的空行
    text = re.sub(r'\n\s*\n', '\n', text)
    # 去除首尾空白
    text = text.strip()
    return text

@app.route('/detect_face', methods=['POST'])
def detect_face():
    try:
        data = request.json
        image_data = data.get('image', '')
        
        if not image_data:
            return jsonify({"error": "图片数据不能为空"}), 400

        # 将base64图片数据转换为numpy数组
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "无法解码图片数据"}), 400

        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用dlib检测人脸
        faces = face_detector(gray)
        
        if len(faces) == 0:
            return jsonify({"error": "未检测到人脸"}), 400
        
        # 获取第一个检测到的人脸
        face = faces[0]
        
        # 获取人脸关键点
        landmarks = landmark_predictor(gray, face)

        # 使用DeepFace进行表情识别
        try:
            result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            
            # 将英文表情转换为中文
            emotion_map = {
                'angry': '生气',
                'disgust': '厌恶',
                'fear': '恐惧',
                'happy': '开心',
                'sad': '悲伤',
                'surprise': '惊讶',
                'neutral': '平静'
            }
            
            emotion_cn = emotion_map.get(emotion, emotion)
            
            # 将 numpy float32 转换为 Python float
            confidence = float(result[0]['emotion'][emotion])
            
            return jsonify({
                "result": emotion_cn,
                "confidence": confidence
            })
            
        except Exception as e:
            logger.error(f"表情识别失败: {str(e)}")
            return jsonify({"error": "表情识别失败"}), 500

    except Exception as e:
        logger.error(f"表情识别失败: {str(e)}")
        return jsonify({"error": "表情识别失败"}), 500

@app.route('/get_models', methods=['GET'])
def get_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = [model['name'] for model in response.json()['models']]
            return jsonify({"models": models})
        else:
            return jsonify({"error": "获取模型列表失败"}), 500
    except Exception as e:
        logger.error(f"获取模型列表时出错: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '')
        emotion = data.get('emotion', '')
        model = data.get('model', 'llama2')
        api_type = data.get('api_type', 'ollama')  # 新增参数，前端可传 'ollama' 或 'deepseek'

        if not message:
            return jsonify({"error": "消息不能为空"}), 400

        # 构建提示词
        prompt = f"你是一个心理咨询师，乐于倾听用户的心声，用户表情状态: {emotion}\n用户消息: {message}\n请根据用户的表情状态和消息内容，给出合适的建议。"

        if api_type == 'deepseek':
            logger.info('走 deepseek 分支')
            logger.info(f"deepseek分支收到的模型名: {model}")
            deepseek_url = 'https://api.deepseek.com/v1/chat/completions'
            api_key = data.get('api_key', 'YOUR_DEEPSEEK_API_KEY')

            # 只允许官方支持的模型名
            allowed_models = ['deepseek-chat', 'deepseek-coder']
            if model not in allowed_models:
                logger.error(f"不支持的 deepseek 模型名: {model}")
                return jsonify({"error": f"不支持的 deepseek 模型名: {model}"}), 400

            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "你是一个心理咨询师，乐于倾听用户的心声。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2000
            }
            response = requests.post(deepseek_url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    reply = result['choices'][0]['message']['content']
                    return jsonify({"response": reply})
                else:
                    return jsonify({"error": "deepseek响应格式错误"}), 500
            else:
                logger.error(f"deepseek API错误: {response.text}")
                return jsonify({"error": f"deepseek API错误: {response.text}"}), 500
        else:
            logger.info('走 ollama 分支')
            logger.info(f"发送到Ollama的请求: model={model}, prompt={prompt}")
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 2000
                    }
                }
            )
            logger.info(f"Ollama API响应状态码: {response.status_code}")
            logger.info(f"Ollama API响应内容: {response.text}")
            if response.status_code == 200:
                result = response.json()
                if 'response' in result:
                    cleaned_response = clean_response(result['response'])
                    logger.info(f"清理后的响应: {cleaned_response}")
                    return jsonify({"response": cleaned_response})
                else:
                    logger.error(f"Ollama API响应中没有response字段: {result}")
                    return jsonify({"error": "模型响应格式错误"}), 500
            else:
                logger.error(f"Ollama API错误: {response.text}")
                return jsonify({"error": "模型响应失败"}), 500

    except Exception as e:
        logger.error(f"处理聊天请求时出错: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)


