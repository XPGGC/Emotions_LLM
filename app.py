from flask import Flask, request, jsonify, Response, stream_with_context
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
from datetime import datetime
import uuid
import os
import time
import hashlib

app = Flask(__name__)
CORS(app)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama API 配置
OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "deepseek-r1:7b"

# 会话存储 - 用于保存对话历史
sessions = {}

# TTS服务配置
TTS_SERVICES = {
    'azure': {
        'enabled': True,  # 改为True，让用户可以配置
        'api_key': os.getenv('AZURE_SPEECH_KEY', ''),
        'region': os.getenv('AZURE_SPEECH_REGION', ''),
        'voices': {
            'zh-CN-XiaoxiaoNeural': '晓晓 (女声，最自然)',
            'zh-CN-YunxiNeural': '云希 (男声，温和专业)',
            'zh-CN-YunyangNeural': '云扬 (男声，成熟稳重)',
            'zh-CN-XiaoyiNeural': '晓伊 (女声，清晰悦耳)',
            'zh-CN-YunjianNeural': '云健 (男声，专业可靠)',
            'zh-CN-XiaohanNeural': '晓涵 (女声，甜美亲切)',
            'zh-CN-XiaomoNeural': '晓墨 (女声，知性优雅)',
            'zh-CN-XiaoxuanNeural': '晓萱 (女声，活泼可爱)',
            'zh-CN-XiaoyanNeural': '晓颜 (女声，温柔细腻)',
            'zh-CN-YunfengNeural': '云枫 (男声，年轻活力)',
            'zh-CN-YunhaoNeural': '云皓 (男声，阳光开朗)',
            'zh-CN-YunjiaNeural': '云嘉 (男声，亲和力强)',
            'zh-CN-YunzeNeural': '云泽 (男声，深沉有力)',
        }
    },
    'aliyun': {
        'enabled': True,  # 改为True，让用户可以配置
        'access_key_id': os.getenv('ALIYUN_ACCESS_KEY_ID', ''),
        'access_key_secret': os.getenv('ALIYUN_ACCESS_KEY_SECRET', ''),
        'voices': {
            'xiaoyun': '小云 (女声，自然温和)',
            'xiaogang': '小刚 (男声，专业可靠)',
            'xiaomei': '小美 (女声，甜美亲切)',
            'xiaoyun_emo': '小云情感 (女声，富有感情)',
            'xiaogang_emo': '小刚情感 (男声，富有感情)',
            'xiaomei_emo': '小美情感 (女声，富有感情)',
        }
    },
    'baidu': {
        'enabled': True,  # 改为True，让用户可以配置
        'api_key': os.getenv('BAIDU_API_KEY', ''),
        'secret_key': os.getenv('BAIDU_SECRET_KEY', ''),
        'voices': {
            '0': '度小美 (女声，甜美自然)',
            '1': '度小宇 (男声，温和专业)',
            '3': '度逍遥 (男声，成熟稳重)',
            '4': '度丫丫 (女声，活泼可爱)',
        }
    }
}

# 智能检测API密钥，如果没有配置则禁用相应服务
def check_tts_services():
    """检查TTS服务配置状态"""
    for service_name, service_config in TTS_SERVICES.items():
        if service_name == 'azure':
            if not service_config['api_key'] or not service_config['region']:
                service_config['enabled'] = False
                logger.info(f"Azure TTS服务未配置API密钥，已禁用")
        elif service_name == 'aliyun':
            if not service_config['access_key_id'] or not service_config['access_key_secret']:
                service_config['enabled'] = False
                logger.info(f"阿里云TTS服务未配置API密钥，已禁用")
        elif service_name == 'baidu':
            if not service_config['api_key'] or not service_config['secret_key']:
                service_config['enabled'] = False
                logger.info(f"百度TTS服务未配置API密钥，已禁用")

# 启动时检查TTS服务
check_tts_services()

# 加载dlib的人脸检测器和关键点检测器
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 表情识别相关配置
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# 情感治疗提示词模板
THERAPY_PROMPTS = {
    "default": """你是一位经验丰富的心理咨询师，专门从事情感治疗。你的目标是：

1. **建立信任关系**：以温暖、理解的态度倾听来访者
2. **深度共情**：准确理解来访者的情感状态和内心世界
3. **引导自我觉察**：帮助来访者更好地认识自己的情感和需求
4. **提供支持**：给予情感支持和实用的建议
5. **促进成长**：引导来访者找到解决问题的方法

**治疗原则**：
- 保持专业、温暖、不带评判的态度
- 使用开放式问题引导来访者深入思考
- 适时给予情感确认和验证
- 鼓励来访者表达真实感受
- 提供具体、可操作的建议

**当前来访者状态**：
- 表情状态：{emotion}
- 当前情绪：{current_emotion}
- 对话历史：{conversation_history}

请根据来访者的表情状态、当前情绪和对话历史，给出专业、温暖、有针对性的回应。记住，你的目标是帮助来访者更好地理解自己，找到内心的平静和力量。""",

    "cbt": """你是一位专业的认知行为治疗师(CBT)。你的治疗目标是：

1. **识别认知模式**：帮助来访者识别自动思维和认知扭曲
2. **挑战不合理信念**：引导来访者质疑和重新评估自己的想法
3. **行为激活**：鼓励来访者采取积极的行动
4. **情绪调节**：教授情绪管理技巧

**CBT技术**：
- 使用苏格拉底式提问
- 引导认知重构
- 教授放松技巧
- 设定行为目标

**当前来访者状态**：
- 表情状态：{emotion}
- 当前情绪：{current_emotion}
- 对话历史：{conversation_history}

请运用CBT技术，帮助来访者识别和改变不健康的思维模式。""",

    "mindfulness": """你是一位正念治疗师，专注于：

1. **当下觉察**：引导来访者关注此时此刻的体验
2. **非评判态度**：培养对内在体验的接纳态度
3. **呼吸觉察**：教授呼吸觉察技巧
4. **情绪觉察**：帮助来访者观察情绪而不被情绪控制

**正念练习**：
- 引导呼吸觉察
- 身体扫描练习
- 情绪觉察练习
- 慈悲冥想

**当前来访者状态**：
- 表情状态：{emotion}
- 当前情绪：{current_emotion}
- 对话历史：{conversation_history}

请运用正念技术，帮助来访者培养内在的平静和觉察能力。"""
}

def get_therapy_prompt(therapy_type, emotion, current_emotion, conversation_history):
    """根据治疗类型获取相应的提示词"""
    prompt_template = THERAPY_PROMPTS.get(therapy_type, THERAPY_PROMPTS["default"])
    return prompt_template.format(
        emotion=emotion,
        current_emotion=current_emotion,
        conversation_history=conversation_history
    )

def analyze_emotional_state(emotion, message):
    """分析情感状态，结合表情和文字内容"""
    # 情感关键词映射
    emotion_keywords = {
        'angry': ['生气', '愤怒', '恼火', '烦躁', '不满', '愤怒', '暴躁'],
        'sad': ['悲伤', '难过', '伤心', '沮丧', '失落', '绝望', '痛苦'],
        'fear': ['害怕', '恐惧', '担心', '焦虑', '紧张', '不安', '恐慌'],
        'happy': ['开心', '快乐', '高兴', '兴奋', '愉悦', '满足', '幸福'],
        'surprise': ['惊讶', '震惊', '意外', '吃惊', '诧异'],
        'disgust': ['厌恶', '恶心', '反感', '讨厌', '嫌弃'],
        'neutral': ['平静', '正常', '一般', '还好', '普通']
    }
    
    # 分析文字中的情感关键词
    text_emotion = 'neutral'
    max_count = 0
    
    for emotion_type, keywords in emotion_keywords.items():
        count = sum(1 for keyword in keywords if keyword in message)
        if count > max_count:
            max_count = count
            text_emotion = emotion_type
    
    # 结合表情和文字分析
    if emotion and text_emotion != 'neutral':
        # 如果表情和文字情感一致，增强该情感
        if emotion == text_emotion:
            return f"强烈{emotion}"
        else:
            # 如果不一致，可能需要更深入的分析
            return f"复杂情感状态：表情显示{emotion}，但言语表达{text_emotion}"
    elif emotion:
        return emotion
    elif text_emotion != 'neutral':
        return text_emotion
    else:
        return "平静"

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

# TTS服务函数
def get_azure_tts(text, voice_name, api_key, region):
    """Azure TTS服务"""
    try:
        url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
        headers = {
            'Ocp-Apim-Subscription-Key': api_key,
            'Content-Type': 'application/ssml+xml',
            'X-Microsoft-OutputFormat': 'audio-16khz-128kbitrate-mono-mp3'
        }
        
        ssml = f"""
        <speak version='1.0' xml:lang='zh-CN'>
            <voice xml:lang='zh-CN' xml:gender='Female' name='{voice_name}'>
                <prosody rate="0.9" pitch="+0%" volume="+0%">
                    {text}
                </prosody>
            </voice>
        </speak>
        """
        
        response = requests.post(url, headers=headers, data=ssml.encode('utf-8'))
        if response.status_code == 200:
            return response.content
        else:
            logger.error(f"Azure TTS错误: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Azure TTS异常: {str(e)}")
        return None

def get_aliyun_tts(text, voice_name, access_key_id, access_key_secret):
    """阿里云TTS服务"""
    try:
        import base64
        import hmac
        import hashlib
        from datetime import datetime
        import urllib.parse
        
        # 阿里云TTS API配置
        url = "https://nls-gateway.cn-shanghai.aliyuncs.com/stream/v1/tts"
        
        # 构建请求参数
        params = {
            'format': 'mp3',
            'voice': voice_name,
            'volume': 50,
            'sample_rate': 16000,
            'speech_rate': 0,
            'pitch_rate': 0,
            'text': text
        }
        
        # 构建签名
        timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        string_to_sign = f"GET\n\n\n{timestamp}\n/stream/v1/tts"
        
        signature = base64.b64encode(
            hmac.new(
                access_key_secret.encode('utf-8'),
                string_to_sign.encode('utf-8'),
                hashlib.sha1
            ).digest()
        ).decode('utf-8')
        
        headers = {
            'Authorization': f'Dataplus {access_key_id}:{signature}',
            'Date': timestamp,
            'Content-Type': 'application/json'
        }
        
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            return response.content
        else:
            logger.error(f"阿里云TTS错误: {response.text}")
            return None
    except Exception as e:
        logger.error(f"阿里云TTS异常: {str(e)}")
        return None

def get_baidu_tts(text, voice_name, api_key, secret_key):
    """百度TTS服务"""
    try:
        # 获取access token
        token_url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}"
        token_response = requests.post(token_url)
        
        if token_response.status_code != 200:
            logger.error(f"百度TTS获取token失败: {token_response.text}")
            return None
            
        access_token = token_response.json().get('access_token')
        
        # TTS API
        tts_url = "https://tsn.baidu.com/text2audio"
        params = {
            'tex': text,
            'tok': access_token,
            'cuid': 'emotions_llm',
            'ctp': 1,
            'lan': 'zh',
            'spd': 5,  # 语速
            'pit': 5,  # 音调
            'vol': 5,  # 音量
            'per': voice_name,  # 发音人
            'aue': 3  # 格式
        }
        
        response = requests.post(tts_url, data=params)
        if response.status_code == 200:
            return response.content
        else:
            logger.error(f"百度TTS错误: {response.text}")
            return None
    except Exception as e:
        logger.error(f"百度TTS异常: {str(e)}")
        return None

@app.route('/tts', methods=['POST'])
def text_to_speech():
    """文本转语音API"""
    try:
        data = request.json
        text = data.get('text', '')
        service = data.get('service', 'azure')  # azure, aliyun, baidu
        voice_name = data.get('voice_name', 'zh-CN-XiaoxiaoNeural')
        
        if not text:
            return jsonify({"error": "文本不能为空"}), 400
        
        audio_data = None
        
        if service == 'azure' and TTS_SERVICES['azure']['enabled']:
            audio_data = get_azure_tts(
                text, 
                voice_name, 
                TTS_SERVICES['azure']['api_key'], 
                TTS_SERVICES['azure']['region']
            )
        elif service == 'aliyun' and TTS_SERVICES['aliyun']['enabled']:
            audio_data = get_aliyun_tts(
                text, 
                voice_name, 
                TTS_SERVICES['aliyun']['access_key_id'], 
                TTS_SERVICES['aliyun']['access_key_secret']
            )
        elif service == 'baidu' and TTS_SERVICES['baidu']['enabled']:
            audio_data = get_baidu_tts(
                text, 
                voice_name, 
                TTS_SERVICES['baidu']['api_key'], 
                TTS_SERVICES['baidu']['secret_key']
            )
        
        if audio_data:
            # 将音频数据编码为base64
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            return jsonify({
                "audio": f"data:audio/mp3;base64,{audio_base64}",
                "service": service,
                "voice": voice_name
            })
        else:
            return jsonify({"error": "TTS服务暂时不可用"}), 500
            
    except Exception as e:
        logger.error(f"TTS服务失败: {str(e)}")
        return jsonify({"error": "TTS服务失败"}), 500

@app.route('/tts_services', methods=['GET'])
def get_tts_services():
    """获取可用的TTS服务"""
    try:
        services_info = {}
        for service_name, service_config in TTS_SERVICES.items():
            services_info[service_name] = {
                'enabled': service_config['enabled'],
                'voices': service_config['voices']
            }
        return jsonify(services_info)
    except Exception as e:
        logger.error(f"获取TTS服务信息失败: {str(e)}")
        return jsonify({"error": "获取TTS服务信息失败"}), 500

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

@app.route('/chat_stream', methods=['POST'])
def chat_stream():
    try:
        data = request.json
        message = data.get('message', '')
        emotion = data.get('emotion', '')
        model = data.get('model', 'llama2')
        api_type = data.get('api_type', 'ollama')
        
        if not message:
            return jsonify({"error": "消息不能为空"}), 400

        prompt = f"你是一个心理咨询师，乐于倾听用户的心声，用户表情状态: {emotion}\n用户消息: {message}\n请根据用户的表情状态和消息内容，给出合适的建议。"

        def ollama_stream():
            try:
                response = requests.post(
                    OLLAMA_API_URL,
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": True,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "max_tokens": 2000
                        }
                    },
                    stream=True
                )
                filter_keywords = [
                    '思考', '推理', '分析', '让我们一步一步来', '首先', '接下来', '让我们思考一下',
                    '我们可以得出', '我们可以推断', '我们可以分析', '我们可以看到', '我们可以认为',
                    '我们可以假设', '我们可以尝试', '我们可以推测', '我们可以总结',
                    '让我们来', '接下来让我们', '下面让我们', '接下来分析', '首先分析', '首先我们',
                    '思路', '推断', '步骤', '过程', '推测', '假设', '分析如下', '推理如下', '思考如下'
                ]
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode())
                            content = chunk.get('response', '')
                            if content:
                                # 过滤掉<think>...</think>标签及其内容，以及单独的<think>或</think>标签
                                content = re.sub(r'<think>[\s\S]*?</think>', '', content)
                                content = re.sub(r'</?think>', '', content)
                                # 过滤关键词
                                if any(kw in content for kw in filter_keywords):
                                    continue
                                # 内容为空或只剩空白标签则不输出
                                if content.strip():
                                    yield content
                        except Exception as e:
                            continue
            except Exception as e:
                logger.error(f"Ollama 流式输出失败: {str(e)}")
                yield '\n[流式输出异常]'

        def deepseek_stream():
            try:
                deepseek_url = 'https://api.deepseek.com/v1/chat/completions'
                api_key = data.get('api_key', 'YOUR_DEEPSEEK_API_KEY')
                allowed_models = ['deepseek-chat', 'deepseek-coder']
                if model not in allowed_models:
                    yield f"[不支持的 deepseek 模型名: {model}]"
                    return
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
                    "max_tokens": 2000,
                    "stream": True
                }
                response = requests.post(deepseek_url, headers=headers, json=payload, stream=True)
                for line in response.iter_lines():
                    if line and line.startswith(b"data: "):
                        data_str = line[6:].decode()
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk['choices'][0]['delta']
                            content = delta.get('content', '')
                            if content:
                                yield content
                        except Exception as e:
                            continue
            except Exception as e:
                logger.error(f"Deepseek 流式输出失败: {str(e)}")
                yield '\n[流式输出异常]'

        if api_type == 'deepseek':
            return Response(stream_with_context(deepseek_stream()), mimetype='text/plain')
        else:
            return Response(stream_with_context(ollama_stream()), mimetype='text/plain')
    except Exception as e:
        logger.error(f"处理流式聊天请求时出错: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/start_session', methods=['POST'])
def start_session():
    """开始新的治疗会话"""
    try:
        data = request.json
        therapy_type = data.get('therapy_type', 'default')
        initial_emotion = data.get('emotion', '')
        
        # 生成会话ID
        session_id = str(uuid.uuid4())
        
        # 创建新会话
        sessions[session_id] = {
            'id': session_id,
            'therapy_type': therapy_type,
            'start_time': datetime.now().isoformat(),
            'conversation_history': [],
            'current_emotion': initial_emotion,
            'session_data': {
                'mood_tracking': [],
                'goals': [],
                'insights': []
            }
        }
        
        # 生成欢迎消息
        welcome_messages = {
            'default': "你好！我是你的心理咨询师。我注意到你现在的心情状态，请告诉我你最近遇到了什么困扰，我会在这里倾听和支持你。",
            'cbt': "你好！我是你的认知行为治疗师。我们可以一起探索你的思维模式，找到更健康的思考方式。请告诉我你最近的想法和感受。",
            'mindfulness': "你好！我是你的正念治疗师。让我们一起培养内在的觉察和平静。请告诉我你现在的感受，我们可以从呼吸觉察开始。"
        }
        
        welcome_msg = welcome_messages.get(therapy_type, welcome_messages['default'])
        
        # 添加欢迎消息到历史记录
        sessions[session_id]['conversation_history'].append({
            'role': 'assistant',
            'content': welcome_msg,
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            'session_id': session_id,
            'welcome_message': welcome_msg,
            'therapy_type': therapy_type
        })
        
    except Exception as e:
        logger.error(f"创建会话失败: {str(e)}")
        return jsonify({"error": "创建会话失败"}), 500

@app.route('/chat_therapy', methods=['POST'])
def chat_therapy():
    """改进的治疗对话API"""
    try:
        data = request.json
        session_id = data.get('session_id', '')
        message = data.get('message', '')
        emotion = data.get('emotion', '')
        model = data.get('model', 'llama2')
        api_type = data.get('api_type', 'ollama')
        
        if not message:
            return jsonify({"error": "消息不能为空"}), 400
            
        if not session_id or session_id not in sessions:
            return jsonify({"error": "无效的会话ID"}), 400
        
        session = sessions[session_id]
        
        # 分析情感状态
        current_emotion = analyze_emotional_state(emotion, message)
        session['current_emotion'] = current_emotion
        
        # 添加用户消息到历史记录
        session['conversation_history'].append({
            'role': 'user',
            'content': message,
            'emotion': emotion,
            'timestamp': datetime.now().isoformat()
        })
        
        # 构建对话历史字符串（只保留最近的10轮对话）
        recent_history = session['conversation_history'][-20:]  # 保留最近20条消息
        conversation_history = "\n".join([
            f"{'治疗师' if msg['role'] == 'assistant' else '来访者'}: {msg['content']}"
            for msg in recent_history
        ])
        
        # 获取治疗提示词
        therapy_prompt = get_therapy_prompt(
            session['therapy_type'],
            emotion,
            current_emotion,
            conversation_history
        )
        
        # 构建完整的用户消息
        user_message = f"来访者说：{message}\n\n请根据上述治疗原则和来访者的状态，给出专业的回应。"
        
        if api_type == 'deepseek':
            deepseek_url = 'https://api.deepseek.com/v1/chat/completions'
            api_key = data.get('api_key', 'YOUR_DEEPSEEK_API_KEY')
            
            allowed_models = ['deepseek-chat', 'deepseek-coder']
            if model not in allowed_models:
                return jsonify({"error": f"不支持的 deepseek 模型名: {model}"}), 400
            
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": therapy_prompt},
                    {"role": "user", "content": user_message}
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
                    
                    # 添加AI回复到历史记录
                    session['conversation_history'].append({
                        'role': 'assistant',
                        'content': reply,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    return jsonify({"response": reply})
                else:
                    return jsonify({"error": "deepseek响应格式错误"}), 500
            else:
                return jsonify({"error": f"deepseek API错误: {response.text}"}), 500
        else:
            # Ollama API
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": model,
                    "prompt": f"{therapy_prompt}\n\n{user_message}",
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 2000
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'response' in result:
                    reply = clean_response(result['response'])
                    
                    # 添加AI回复到历史记录
                    session['conversation_history'].append({
                        'role': 'assistant',
                        'content': reply,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    return jsonify({"response": reply})
                else:
                    return jsonify({"error": "模型响应格式错误"}), 500
            else:
                return jsonify({"error": "模型响应失败"}), 500
                
    except Exception as e:
        logger.error(f"治疗对话失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/chat_therapy_stream', methods=['POST'])
def chat_therapy_stream():
    """流式治疗对话API"""
    try:
        data = request.json
        session_id = data.get('session_id', '')
        message = data.get('message', '')
        emotion = data.get('emotion', '')
        model = data.get('model', 'llama2')
        api_type = data.get('api_type', 'ollama')
        
        if not message:
            return jsonify({"error": "消息不能为空"}), 400
            
        if not session_id or session_id not in sessions:
            return jsonify({"error": "无效的会话ID"}), 400
        
        session = sessions[session_id]
        
        # 分析情感状态
        current_emotion = analyze_emotional_state(emotion, message)
        session['current_emotion'] = current_emotion
        
        # 添加用户消息到历史记录
        session['conversation_history'].append({
            'role': 'user',
            'content': message,
            'emotion': emotion,
            'timestamp': datetime.now().isoformat()
        })
        
        # 构建对话历史字符串
        recent_history = session['conversation_history'][-20:]
        conversation_history = "\n".join([
            f"{'治疗师' if msg['role'] == 'assistant' else '来访者'}: {msg['content']}"
            for msg in recent_history
        ])
        
        # 获取治疗提示词
        therapy_prompt = get_therapy_prompt(
            session['therapy_type'],
            emotion,
            current_emotion,
            conversation_history
        )
        
        user_message = f"来访者说：{message}\n\n请根据上述治疗原则和来访者的状态，给出专业的回应。"
        
        def ollama_therapy_stream():
            try:
                response = requests.post(
                    OLLAMA_API_URL,
                    json={
                        "model": model,
                        "prompt": f"{therapy_prompt}\n\n{user_message}",
                        "stream": True,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "max_tokens": 2000
                        }
                    },
                    stream=True
                )
                
                full_response = ""
                filter_keywords = [
                    '思考', '推理', '分析', '让我们一步一步来', '首先', '接下来', '让我们思考一下',
                    '我们可以得出', '我们可以推断', '我们可以分析', '我们可以看到', '我们可以认为',
                    '我们可以假设', '我们可以尝试', '我们可以推测', '我们可以总结',
                    '让我们来', '接下来让我们', '下面让我们', '接下来分析', '首先分析', '首先我们',
                    '思路', '推断', '步骤', '过程', '推测', '假设', '分析如下', '推理如下', '思考如下'
                ]
                
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode())
                            content = chunk.get('response', '')
                            if content:
                                # 过滤掉<think>...</think>标签及其内容
                                content = re.sub(r'<think>[\s\S]*?</think>', '', content)
                                content = re.sub(r'</?think>', '', content)
                                # 过滤关键词
                                if any(kw in content for kw in filter_keywords):
                                    continue
                                # 内容为空或只剩空白标签则不输出
                                if content.strip():
                                    full_response += content
                                    yield content
                        except Exception as e:
                            continue
                
                # 流式输出完成后，将完整回复添加到历史记录
                if full_response.strip():
                    session['conversation_history'].append({
                        'role': 'assistant',
                        'content': full_response.strip(),
                        'timestamp': datetime.now().isoformat()
                    })
                    
            except Exception as e:
                logger.error(f"Ollama 治疗流式输出失败: {str(e)}")
                yield '\n[流式输出异常]'

        def deepseek_therapy_stream():
            try:
                deepseek_url = 'https://api.deepseek.com/v1/chat/completions'
                api_key = data.get('api_key', 'YOUR_DEEPSEEK_API_KEY')
                allowed_models = ['deepseek-chat', 'deepseek-coder']
                
                if model not in allowed_models:
                    yield f"[不支持的 deepseek 模型名: {model}]"
                    return
                
                headers = {
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                }
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": therapy_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 2000,
                    "stream": True
                }
                
                response = requests.post(deepseek_url, headers=headers, json=payload, stream=True)
                full_response = ""
                
                for line in response.iter_lines():
                    if line and line.startswith(b"data: "):
                        data_str = line[6:].decode()
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk['choices'][0]['delta']
                            content = delta.get('content', '')
                            if content:
                                full_response += content
                                yield content
                        except Exception as e:
                            continue
                
                # 流式输出完成后，将完整回复添加到历史记录
                if full_response.strip():
                    session['conversation_history'].append({
                        'role': 'assistant',
                        'content': full_response.strip(),
                        'timestamp': datetime.now().isoformat()
                    })
                    
            except Exception as e:
                logger.error(f"Deepseek 治疗流式输出失败: {str(e)}")
                yield '\n[流式输出异常]'

        if api_type == 'deepseek':
            return Response(stream_with_context(deepseek_therapy_stream()), mimetype='text/plain')
        else:
            return Response(stream_with_context(ollama_therapy_stream()), mimetype='text/plain')
            
    except Exception as e:
        logger.error(f"流式治疗对话失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/session/<session_id>', methods=['GET'])
def get_session(session_id):
    """获取会话信息"""
    try:
        if session_id not in sessions:
            return jsonify({"error": "会话不存在"}), 404
        
        session = sessions[session_id]
        return jsonify({
            'session_id': session_id,
            'therapy_type': session['therapy_type'],
            'start_time': session['start_time'],
            'current_emotion': session['current_emotion'],
            'conversation_count': len(session['conversation_history']),
            'session_data': session['session_data']
        })
        
    except Exception as e:
        logger.error(f"获取会话信息失败: {str(e)}")
        return jsonify({"error": "获取会话信息失败"}), 500

@app.route('/session/<session_id>/history', methods=['GET'])
def get_session_history(session_id):
    """获取会话历史"""
    try:
        if session_id not in sessions:
            return jsonify({"error": "会话不存在"}), 404
        
        session = sessions[session_id]
        return jsonify({
            'session_id': session_id,
            'conversation_history': session['conversation_history']
        })
        
    except Exception as e:
        logger.error(f"获取会话历史失败: {str(e)}")
        return jsonify({"error": "获取会话历史失败"}), 500

@app.route('/session/<session_id>', methods=['DELETE'])
def end_session(session_id):
    """结束会话"""
    try:
        if session_id not in sessions:
            return jsonify({"error": "会话不存在"}), 404
        
        session = sessions.pop(session_id)
        
        # 生成会话总结
        summary = f"会话结束。共进行了{len(session['conversation_history'])}轮对话。"
        if session['session_data']['insights']:
            summary += f" 主要洞察：{', '.join(session['session_data']['insights'][-3:])}"
        
        return jsonify({
            'message': '会话已结束',
            'summary': summary,
            'session_duration': len(session['conversation_history'])
        })
        
    except Exception as e:
        logger.error(f"结束会话失败: {str(e)}")
        return jsonify({"error": "结束会话失败"}), 500

if __name__ == '__main__':
    app.run(debug=False)


