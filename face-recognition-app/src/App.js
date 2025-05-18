import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
    const [image, setImage] = useState(null);
    const [emotionResult, setEmotionResult] = useState('');
    const [message, setMessage] = useState('');
    const [chatResponse, setChatResponse] = useState('');
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const handleImageChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onloadend = () => {
                setImage(reader.result);
            };
            reader.readAsDataURL(file);
        }
    };

    const handleDetectFace = async () => {
        if (!image || isLoading) return;
        setIsLoading(true);
        setError('');
        try {
            const response = await axios.post('http://127.0.0.1:5000/detect_face', { image });
            setEmotionResult(response.data.result);
        } catch (error) {
            setError('表情检测请求失败，请检查控制台日志');
        } finally {
            setIsLoading(false);
        }
    };

    const handleChat = async () => {
        if (!message.trim() || isLoading) return;
        setIsLoading(true);
        setError('');
        try {
            const response = await axios.post('http://127.0.0.1:5000/chat', { 
                message, 
                emotion: emotionResult 
            });
            setChatResponse(response.data.response);
            setMessage(''); 
        } catch (error) {
            setError('与大语言模型交互请求失败，请检查控制台日志');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="app-container">
            <h1 className="title">智能交互助手</h1>
            
            <div className="module-card">
                <h2 className="module-title">人脸表情识别</h2>
                <div className="upload-section">
                    <label className="custom-file-upload">
                        {image ? "已选择图片" : "点击选择图片"}
                        <input 
                            type="file" 
                            onChange={handleImageChange} 
                            accept="image/*"
                        />
                    </label>
                    {image && (
                        <div className="image-preview">
                            <img 
                                src={image} 
                                alt="预览" 
                                style={{ maxWidth: '300px', maxHeight: '200px' }}
                            />
                        </div>
                    )}
                    <button 
                        onClick={handleDetectFace} 
                        disabled={isLoading || !image}
                        className={isLoading ? 'btn loading' : 'btn primary'}
                    >
                        {isLoading ? '检测中...' : '开始检测表情'}
                    </button>
                </div>
                {emotionResult && (
                    <div className="result-box">
                        <p className="emotion-result">{emotionResult}</p>
                    </div>
                )}
            </div>

            <div className="module-card">
                <h2 className="module-title">智能聊天助手</h2>
                <div className="chat-section">
                    <input
                        type="text"
                        placeholder="输入消息与智能助手对话"
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        disabled={isLoading}
                        className="chat-input"
                    />
                    <button 
                        onClick={handleChat} 
                        disabled={isLoading || !message.trim()}
                        className={isLoading ? 'btn loading' : 'btn secondary'}
                    >
                        {isLoading ? '发送中...' : '发送消息'}
                    </button>
                </div>
                {chatResponse && (
                    <div className="result-box">
                        <p className="chat-response">
                            <span className="response-label">助手回复：</span>
                            {chatResponse}
                        </p>
                    </div>
                )}
            </div>

            {error && <div className="error-message">{error}</div>}
        </div>
    );
}

export default App;