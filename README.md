# 🤖 AI情感治疗系统

一个基于人工智能的情感治疗对话系统，集成了表情识别、语音交互和多种治疗模式，提供专业、温暖的心理支持。

## ✨ 主要功能

### 🎯 智能情感治疗
- **多模式治疗**：一般心理咨询、认知行为治疗(CBT)、正念治疗
- **表情识别**：实时分析用户情绪状态，调整治疗策略
- **个性化对话**：根据用户情况定制专业回复
- **会话管理**：智能记忆对话历史，提供连贯的治疗体验

### 🎤 高级语音交互
- **多TTS服务**：支持Azure、阿里云、百度等高质量语音服务
- **语音识别**：支持语音输入，解放双手
- **自然语音**：接近豆包的人声效果，温暖自然
- **语音优化**：智能参数调整，获得最佳听觉体验

### 🧠 AI模型支持
- **多模型切换**：支持Ollama、DeepSeek等多种LLM模型
- **流式对话**：实时响应，提升交互体验
- **智能分析**：深度理解用户情感和需求

## 🚀 快速开始

### 环境要求
- Python 3.8+
- Node.js 14+
- 麦克风设备（语音功能）

### 配置文件设置
1. 复制示例配置文件：
```bash
cp config.env.example config.env
```

2. 编辑 `config.env` 文件，填入您的API密钥（可选）

### 一键启动
```bash
# Windows用户
start_therapy.bat

# 或手动启动
python app.py
cd face-recognition-app && npm start
```

### 访问系统
- 前端界面：http://localhost:3000
- 后端API：http://127.0.0.1:5000

## 🎤 语音功能配置

### 浏览器内置语音（免费）
系统默认使用浏览器内置语音，无需额外配置。

### 高级TTS服务（推荐）
获得更接近豆包的自然人声效果：

#### Azure语音服务（最佳效果）
1. 注册Azure账户：https://azure.microsoft.com/
2. 创建语音服务资源
3. 获取API Key和Region
4. 在`config.env`中配置：
```
AZURE_SPEECH_KEY=your_api_key_here
AZURE_SPEECH_REGION=your_region_here
```

#### 百度语音服务
1. 注册百度智能云账户
2. 创建语音应用
3. 获取API Key和Secret Key
4. 在`config.env`中配置：
```
BAIDU_API_KEY=your_api_key
BAIDU_SECRET_KEY=your_secret_key
```

### 使用步骤
1. 配置TTS服务API密钥
2. 重启系统：`start_therapy.bat`
3. 点击"⚙️ 语音设置"
4. 选择TTS服务和语音
5. 测试并启用语音功能

## 📁 项目结构

```
Emotions_LLM/
├── app.py                          # Flask后端主程序
├── requirements.txt                # Python依赖
├── start_therapy.bat              # 一键启动脚本
├── config.env                     # 环境变量配置
├── facial_expression_model_weights.h5  # 表情识别模型
├── shape_predictor_68_face_landmarks.dat  # 人脸关键点模型
├── face-recognition-app/          # React前端
│   ├── src/
│   │   ├── App.js                 # 主应用组件
│   │   ├── App.css                # 样式文件
│   │   ├── index.js               # 入口文件
│   │   └── index.css              # 全局样式
│   ├── public/
│   │   └── index.html             # HTML模板
│   └── package.json               # 前端依赖
└── images/                        # 图片资源
```

## 🎯 使用指南

### 1. 开始治疗
1. 启动系统后，浏览器自动打开界面
2. 选择治疗模式（一般咨询/CBT/正念）
3. 上传照片进行表情识别
4. 开始与AI治疗师对话

### 2. 语音功能
1. 点击"⚙️ 语音设置"
2. 选择TTS服务（浏览器/Azure/百度）
3. 选择语音并调整参数
4. 启用语音播放和语音输入
5. 使用🎤按钮进行语音输入

### 3. 表情识别
1. 允许浏览器访问摄像头或上传照片
2. 系统自动检测表情和情绪
3. AI根据表情变化调整回复策略

## 🔧 配置说明

### 模型配置
在`app.py`中修改默认模型：
```python
DEFAULT_MODEL = "deepseek-r1:7b"  # 默认模型
```

### API配置
- **Ollama**：本地运行，无需API密钥
- **DeepSeek**：需要设置API密钥
- **TTS服务**：在`config.env`中配置

## 🛠️ 故障排除

### 常见问题

1. **系统无法启动**
   - 检查Python和Node.js版本
   - 确认依赖包安装完整
   - 检查端口是否被占用

2. **语音功能异常**
   - 确认浏览器支持语音API
   - 检查麦克风权限设置
   - 验证TTS服务配置

3. **表情识别失败**
   - 确认摄像头权限
   - 检查模型文件是否存在
   - 确保光线充足

4. **对话质量差**
   - 尝试不同的模型
   - 调整治疗模式
   - 检查网络连接

## 📄 许可证

本项目采用 MIT 许可证。

## 🤝 贡献

欢迎提交Issue和Pull Request！

## ⚠️ 重要声明

**本系统仅供学习和研究使用，不能替代专业心理治疗。如有严重的心理健康问题，请寻求专业帮助。**

---

**让AI陪伴您，温暖您的心灵** 💙
