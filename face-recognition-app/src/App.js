import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

function App() {
    const [image, setImage] = useState(null);
    const [emotionResult, setEmotionResult] = useState('');
    const [message, setMessage] = useState('');
    const [chatResponse, setChatResponse] = useState('');
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [ollamaModels, setOllamaModels] = useState([]);
    const [deepseekModels] = useState(['deepseek-chat']);
    const [selectedModel, setSelectedModel] = useState('deepseek-chat');
    const [chatHistory, setChatHistory] = useState([]);
    const messagesEndRef = useRef(null);
    const [apiType, setApiType] = useState('ollama');
    const apiTypeRef = useRef(apiType);
    const [deepseekApiKey, setDeepseekApiKey] = useState('');

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [chatHistory]);

    useEffect(() => { apiTypeRef.current = apiType; }, [apiType]);

    useEffect(() => {
        if (apiType === 'ollama') {
            const fetchModels = async () => {
                try {
                    const response = await axios.get('http://127.0.0.1:5000/get_models');
                    setOllamaModels(response.data.models);
                    if (response.data.models.length > 0) {
                        setSelectedModel(response.data.models[0]);
                    }
                } catch (error) {
                    setError('获取模型列表失败，请检查Ollama服务是否运行');
                }
            };
            fetchModels();
        } else if (apiType === 'deepseek') {
            setSelectedModel('deepseek-chat');
        }
    }, [apiType]);

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
        
        // 添加用户消息到历史记录
        const userMessage = { type: 'user', content: message };
        setChatHistory(prev => [...prev, userMessage]);
        
        try {
            const requestBody = { 
                message, 
                emotion: emotionResult,
                model: selectedModel,
                api_type: apiType
            };
            if (apiType === 'deepseek') {
                requestBody.api_key = deepseekApiKey;
            }
            const response = await axios.post('http://127.0.0.1:5000/chat', requestBody);
            
            console.log('API响应:', response.data); // 添加日志
            
            if (response.data.response) {
                // 添加助手回复到历史记录
                const assistantMessage = { type: 'assistant', content: response.data.response };
                setChatHistory(prev => [...prev, assistantMessage]);
            } else {
                setError('模型返回的响应为空');
            }
            
            setMessage(''); 
        } catch (error) {
            console.error('聊天请求错误:', error); // 添加错误日志
            setError('与大语言模型交互请求失败，请检查控制台日志');
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleChat();
        }
    };

    return (
        <div className="app-container">
            <div className="sidebar">
                <div className="sidebar-header">
                    <h1 className="app-title">智能交互助手</h1>
                </div>
                <div className="model-selector">
                    <label htmlFor="model-select">选择模型</label>
                    <select
                        id="model-select"
                        value={selectedModel}
                        onChange={(e) => {
                            if (apiType === 'ollama') setSelectedModel(e.target.value);
                        }}
                        className="model-select"
                        disabled={apiType === 'deepseek'}
                    >
                        {(apiType === 'ollama' ? ollamaModels : deepseekModels).map((model) => (
                            <option key={model} value={model}>
                                {model}
                            </option>
                        ))}
                    </select>
                    <label htmlFor="api-type-select" style={{marginTop: '1rem', display: 'block'}}>选择API</label>
                    <select
                        id="api-type-select"
                        value={apiType}
                        onChange={(e) => setApiType(e.target.value)}
                        className="model-select"
                    >
                        <option value="ollama">Ollama</option>
                        <option value="deepseek">Deepseek</option>
                    </select>
                    {apiType === 'deepseek' && (
                        <div style={{marginTop: '1rem'}}>
                            <label htmlFor="deepseek-api-key">Deepseek API Key</label>
                            <input
                                id="deepseek-api-key"
                                type="text"
                                value={deepseekApiKey}
                                onChange={e => setDeepseekApiKey(e.target.value)}
                                className="model-select"
                                placeholder="请输入你的 Deepseek API Key"
                            />
                        </div>
                    )}
                </div>
                <div className="emotion-section">
                    <h2>表情识别</h2>
                    <div className="upload-section">
                        <div className="image-preview">
                            {image ? (
                                <img 
                                    src={image} 
                                    alt="预览" 
                                />
                            ) : (
                                <div className="upload-placeholder">
                                    <input 
                                        type="file" 
                                        onChange={handleImageChange} 
                                        accept="image/*"
                                        id="image-upload"
                                    />
                                    <label htmlFor="image-upload">
                                        <span>点击或拖拽图片到此处</span>
                                    </label>
                                </div>
                            )}
                        </div>
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
            </div>

            <div className="main-content">
                <div className="chat-container">
                    <div className="chat-messages">
                        {chatHistory.map((msg, index) => (
                            <div key={index} className={`message ${msg.type}`}>
                                <div className="message-content">
                                    {msg.type === 'assistant' && (
                                        <div className="assistant-avatar">AI</div>
                                    )}
                                    <div className="message-text">
                                        {msg.content}
                                    </div>
                                </div>
                            </div>
                        ))}
                        <div ref={messagesEndRef} />
                    </div>
                    <div className="chat-input-wrapper">
                        <div className="chat-input-container">
                            <textarea
                                value={message}
                                onChange={(e) => setMessage(e.target.value)}
                                onKeyPress={handleKeyPress}
                                placeholder="输入消息与智能助手对话..."
                                disabled={isLoading}
                                className="chat-input"
                                rows="1"
                            />
                            <div className="chat-input-actions">
                                <button 
                                    onClick={handleChat} 
                                    disabled={isLoading || !message.trim()}
                                    className={isLoading ? 'send-button loading' : 'send-button'}
                                >
                                    {isLoading ? '' : '发送'}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {error && <div className="error-message">{error}</div>}
        </div>
    );
}

export default App;