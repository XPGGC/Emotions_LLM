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
    const [isStreaming, setIsStreaming] = useState(false);
    
    // 新增状态
    const [sessionId, setSessionId] = useState(null);
    const [therapyType, setTherapyType] = useState('default');
    const [sessionStarted, setSessionStarted] = useState(false);
    const [currentEmotion, setCurrentEmotion] = useState('');
    const [showSessionSetup, setShowSessionSetup] = useState(true);

    // 摄像头相关状态
    const [isCameraOn, setIsCameraOn] = useState(false);
    const [cameraStream, setCameraStream] = useState(null);
    const [videoRef, setVideoRef] = useState(null);
    const [canvasRef, setCanvasRef] = useState(null);

    // 语音相关状态
    const [isSpeaking, setIsSpeaking] = useState(false);
    const [isListening, setIsListening] = useState(false);
    const [speechEnabled, setSpeechEnabled] = useState(true);
    const [voiceEnabled, setVoiceEnabled] = useState(true);
    const [selectedVoice, setSelectedVoice] = useState(null);
    const [availableVoices, setAvailableVoices] = useState([]);
    const [speechRate, setSpeechRate] = useState(0.9);
    const [speechPitch, setSpeechPitch] = useState(1.1);
    const [speechVolume, setSpeechVolume] = useState(0.8);

    // 语音识别相关
    const [recognition, setRecognition] = useState(null);
    const [transcript, setTranscript] = useState('');

    // 高级语音设置
    const [useAdvancedVoice, setUseAdvancedVoice] = useState(false);
    const [voiceStyle, setVoiceStyle] = useState('natural');
    const [pauseBetweenSentences, setPauseBetweenSentences] = useState(true);

    // 高级TTS服务
    const [ttsService, setTtsService] = useState('browser');
    const [ttsServices, setTtsServices] = useState({});
    const [selectedTtsVoice, setSelectedTtsVoice] = useState('');
    const [ttsApiKey, setTtsApiKey] = useState('');
    const [ttsRegion, setTtsRegion] = useState('');

    // 新增状态
    const [showVoiceSettings, setShowVoiceSettings] = useState(false);

    // 添加缺失的ref和函数
    const textareaRef = useRef(null);
    const latestTranscriptRef = useRef(''); // 添加这个ref来存储最新的识别结果

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [chatHistory]);

    useEffect(() => { apiTypeRef.current = apiType; }, [apiType]);

    // 获取TTS服务信息
    useEffect(() => {
        const fetchTtsServices = async () => {
            try {
                const response = await axios.get('http://127.0.0.1:5000/tts_services');
                setTtsServices(response.data);
            } catch (error) {
                console.log('TTS服务信息获取失败，使用浏览器默认语音');
            }
        };
        fetchTtsServices();
    }, []);

    // 初始化语音功能
    useEffect(() => {
        // 初始化语音合成
        if ('speechSynthesis' in window) {
            const loadVoices = () => {
                const voices = speechSynthesis.getVoices();
                if (voices.length > 0) {
                    // 过滤和排序语音，优先选择更自然的语音
                    const filteredVoices = voices.filter(voice => {
                        // 优先选择中文语音，然后是英文语音
                        return voice.lang.includes('zh') || 
                               voice.lang.includes('cmn') || 
                               voice.lang.includes('en');
                    }).sort((a, b) => {
                        // 优先选择中文语音
                        const aIsChinese = a.lang.includes('zh') || a.lang.includes('cmn');
                        const bIsChinese = b.lang.includes('zh') || b.lang.includes('cmn');
                        if (aIsChinese && !bIsChinese) return -1;
                        if (!aIsChinese && bIsChinese) return 1;
                        return 0;
                    });
                    
                    setAvailableVoices(filteredVoices);
                    
                    // 选择最佳语音
                    const bestVoice = filteredVoices.find(voice => 
                        voice.lang.includes('zh') || voice.lang.includes('cmn')
                    ) || filteredVoices[0];
                    
                    setSelectedVoice(bestVoice);
                }
            };
            
            // 立即加载语音
            loadVoices();
            
            // 监听语音列表变化
            speechSynthesis.onvoiceschanged = loadVoices;
        }

        // 初始化语音识别
        const initSpeechRecognition = async () => {
            // 检查浏览器支持
            if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
                console.error('浏览器不支持语音识别功能');
                return;
            }

            // 检查麦克风权限
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                stream.getTracks().forEach(track => track.stop());
                console.log('麦克风权限检查通过');
            } catch (error) {
                console.error('麦克风权限被拒绝:', error);
                return;
            }

            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            const recognitionInstance = new SpeechRecognition();
            
            recognitionInstance.continuous = false;
            recognitionInstance.interimResults = false;
            recognitionInstance.lang = 'zh-CN';
            
            recognitionInstance.onstart = () => {
                console.log('语音识别开始');
                setIsListening(true);
                setTranscript('');
            };
            
            recognitionInstance.onresult = (event) => {
                console.log('语音识别结果:', event.results);
                const transcript = Array.from(event.results)
                    .map(result => result[0])
                    .map(result => result.transcript)
                    .join('');
                console.log('识别文本:', transcript);
                setTranscript(transcript);
                
                // 保存到ref中
                latestTranscriptRef.current = transcript;
                
                // 立即设置到输入框
                if (transcript.trim()) {
                    console.log('立即设置识别文本到输入框:', transcript);
                    setMessage(transcript);
                }
            };
            
            recognitionInstance.onerror = (event) => {
                console.error('语音识别错误:', event.error);
                let errorMessage = '';
                switch(event.error) {
                    case 'not-allowed':
                        errorMessage = '麦克风权限被拒绝';
                        break;
                    case 'no-speech':
                        errorMessage = '没有检测到语音';
                        break;
                    case 'audio-capture':
                        errorMessage = '音频捕获失败';
                        break;
                    case 'network':
                        errorMessage = '网络错误';
                        break;
                    default:
                        errorMessage = `未知错误: ${event.error}`;
                }
                console.error('错误详情:', errorMessage);
                setIsListening(false);
            };
            
            recognitionInstance.onend = () => {
                console.log('语音识别结束');
                setIsListening(false);
                
                // 使用ref中的最新结果
                const finalTranscript = latestTranscriptRef.current;
                if (finalTranscript && finalTranscript.trim()) {
                    console.log('语音识别结束时设置文本到输入框:', finalTranscript);
                    setMessage(finalTranscript);
                }
            };
            
            setRecognition(recognitionInstance);
            console.log('语音识别初始化完成');
        };

        // 初始化语音识别
        initSpeechRecognition();
    }, []);

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

    // 优化文本，使其更适合语音播放
    const optimizeTextForSpeech = (text) => {
        if (!text) return text;
        
        let optimizedText = text;
        
        // 在句子之间添加停顿
        if (pauseBetweenSentences) {
            optimizedText = optimizedText
                .replace(/。/g, '。 ')
                .replace(/！/g, '！ ')
                .replace(/？/g, '？ ')
                .replace(/；/g, '； ')
                .replace(/，/g, '， ')
                .replace(/\./g, '. ')
                .replace(/!/g, '! ')
                .replace(/\?/g, '? ')
                .replace(/;/g, '; ')
                .replace(/,/g, ', ');
        }
        
        // 根据语音风格调整文本
        switch (voiceStyle) {
            case 'warm':
                // 添加温暖的语气词
                optimizedText = optimizedText.replace(/^/, '嗯，');
                break;
            case 'professional':
                // 保持专业语调
                break;
            default:
                // natural - 保持自然
                break;
        }
        
        return optimizedText;
    };

    // 高级TTS服务播放
    const playAdvancedTts = async (text) => {
        try {
            const response = await axios.post('http://127.0.0.1:5000/tts', {
                text: text,
                service: ttsService,
                voice_name: selectedTtsVoice
            });
            
            if (response.data.audio) {
                const audio = new Audio(response.data.audio);
                audio.onplay = () => setIsSpeaking(true);
                audio.onended = () => setIsSpeaking(false);
                audio.onerror = () => setIsSpeaking(false);
                await audio.play();
            }
        } catch (error) {
            console.error('高级TTS播放失败:', error);
            // 回退到浏览器TTS
            speakWithBrowser(text);
        }
    };

    // 浏览器TTS播放
    const speakWithBrowser = (text) => {
        if (!speechEnabled || !selectedVoice) return;
        
        // 停止当前播放
        speechSynthesis.cancel();
        
        // 优化文本
        const optimizedText = optimizeTextForSpeech(text);
        
        const utterance = new SpeechSynthesisUtterance(optimizedText);
        utterance.voice = selectedVoice;
        utterance.rate = speechRate;
        utterance.pitch = speechPitch;
        utterance.volume = speechVolume;
        
        // 设置更自然的语音参数
        if (useAdvancedVoice) {
            utterance.rate = Math.max(0.7, speechRate); // 确保语速不会太快
            utterance.pitch = Math.max(0.8, Math.min(1.3, speechPitch)); // 限制音调范围
        }
        
        utterance.onstart = () => setIsSpeaking(true);
        utterance.onend = () => setIsSpeaking(false);
        utterance.onerror = (event) => {
            console.error('语音合成错误:', event.error);
            setIsSpeaking(false);
        };
        
        speechSynthesis.speak(utterance);
    };

    // 文本转语音功能
    const speak = (text) => {
        if (!speechEnabled) return;
        
        if (ttsService === 'browser') {
            speakWithBrowser(text);
        } else {
            playAdvancedTts(text);
        }
    };

    // 停止语音
    const stopSpeaking = () => {
        speechSynthesis.cancel();
        setIsSpeaking(false);
    };

    // 开始语音识别
    const startListening = () => {
        console.log('尝试开始语音识别...');
        if (!recognition) {
            console.error('语音识别未初始化');
            return;
        }
        
        if (isListening) {
            console.log('语音识别已在运行中');
            return;
        }
        
        try {
            recognition.start();
            console.log('语音识别启动成功');
        } catch (error) {
            console.error('启动语音识别失败:', error);
        }
    };

    // 停止语音识别
    const stopListening = () => {
        console.log('尝试停止语音识别...');
        if (!recognition) {
            console.error('语音识别未初始化');
            return;
        }
        
        if (!isListening) {
            console.log('语音识别未在运行');
            return;
        }
        
        try {
            recognition.stop();
            console.log('语音识别停止成功');
        } catch (error) {
            console.error('停止语音识别失败:', error);
        }
    };

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
            setCurrentEmotion(response.data.result);
        } catch (error) {
            setError('表情检测请求失败，请检查控制台日志');
        } finally {
            setIsLoading(false);
        }
    };

    const startTherapySession = async () => {
        if (isLoading) return;
        setIsLoading(true);
        setError('');
        
        try {
            const response = await axios.post('http://127.0.0.1:5000/start_session', {
                therapy_type: therapyType,
                emotion: emotionResult || currentEmotion
            });
            
            setSessionId(response.data.session_id);
            setSessionStarted(true);
            setShowSessionSetup(false);
            
            // 添加欢迎消息到聊天历史
            const welcomeMessage = response.data.welcome_message;
            setChatHistory([{
                type: 'assistant',
                content: welcomeMessage
            }]);
            
            // 播放欢迎语音
            if (speechEnabled) {
                speak(welcomeMessage);
            }
            
        } catch (error) {
            setError('开始治疗会话失败，请检查控制台日志');
        } finally {
            setIsLoading(false);
        }
    };

    const handleTherapyChat = async () => {
        if (!message.trim() || isLoading || !sessionId) return;
        setIsLoading(true);
        setIsStreaming(true);
        setError('');

        // 添加用户消息到历史记录
        const userMessage = { type: 'user', content: message };
        setChatHistory(prev => [...prev, userMessage]);

        // 先添加一个空的AI消息用于流式填充
        const aiIndex = chatHistory.length + 1;
        setChatHistory(prev => [...prev, { type: 'assistant', content: '' }]);

        try {
            const requestBody = { 
                session_id: sessionId,
                message, 
                emotion: emotionResult || currentEmotion,
                model: selectedModel,
                api_type: apiType
            };
            if (apiType === 'deepseek') {
                requestBody.api_key = deepseekApiKey;
            }
            
            const response = await fetch('http://127.0.0.1:5000/chat_therapy_stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            });
            
            if (!response.body) throw new Error('无流式响应体');
            const reader = response.body.getReader();
            let aiContent = '';
            let done = false;
            
            while (!done) {
                const { value, done: doneReading } = await reader.read();
                done = doneReading;
                if (value) {
                    const chunk = new TextDecoder().decode(value);
                    aiContent += chunk;
                    setChatHistory(prev => {
                        // 只更新最后一条AI消息
                        const updated = [...prev];
                        updated[aiIndex] = { type: 'assistant', content: aiContent };
                        return updated;
                    });
                }
            }
            
            // 流式输出完成后播放语音
            if (speechEnabled && aiContent.trim()) {
                speak(aiContent);
            }
            
            setMessage('');
        } catch (error) {
            setError('治疗对话请求失败，请检查控制台日志');
        } finally {
            setIsLoading(false);
            setIsStreaming(false);
        }
    };

    const endTherapySession = async () => {
        if (!sessionId || isLoading) return;
        setIsLoading(true);
        
        // 停止语音播放
        stopSpeaking();
        
        try {
            await axios.delete(`http://127.0.0.1:5000/session/${sessionId}`);
            setSessionId(null);
            setSessionStarted(false);
            setShowSessionSetup(true);
            setChatHistory([]);
            setCurrentEmotion('');
        } catch (error) {
            setError('结束会话失败');
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (sessionStarted) {
                handleTherapyChat();
            } else {
                handleChat();
            }
        }
    };

    // 输入框自适应高度
    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
        }
    }, [message]);

    // 聊天区自动滚动到底部
    useEffect(() => {
        scrollToBottom();
    }, [chatHistory, isStreaming]);

    // 摄像头相关useEffect
    useEffect(() => {
        // 组件卸载时清理摄像头
        return () => {
            if (cameraStream) {
                cameraStream.getTracks().forEach(track => track.stop());
            }
        };
    }, [cameraStream]);

    // 设置视频源
    useEffect(() => {
        if (videoRef && cameraStream) {
            console.log('useEffect: 设置视频源');
            videoRef.srcObject = cameraStream;
            
            // 添加事件监听
            const video = videoRef;
            
            const handleLoadedMetadata = () => {
                console.log('视频元数据加载完成');
            };
            
            const handleCanPlay = () => {
                console.log('视频可以播放');
                video.play().catch(error => {
                    console.error('自动播放失败:', error);
                });
            };
            
            const handleError = (error) => {
                console.error('视频错误:', error);
                setError('视频加载失败');
            };
            
            video.addEventListener('loadedmetadata', handleLoadedMetadata);
            video.addEventListener('canplay', handleCanPlay);
            video.addEventListener('error', handleError);
            
            // 清理函数
            return () => {
                video.removeEventListener('loadedmetadata', handleLoadedMetadata);
                video.removeEventListener('canplay', handleCanPlay);
                video.removeEventListener('error', handleError);
            };
        }
    }, [videoRef, cameraStream]);

    // 原有的聊天功能（保留兼容性）
    const handleChat = async () => {
        if (!message.trim() || isLoading) return;
        setIsLoading(true);
        setIsStreaming(true);
        setError('');

        // 添加用户消息到历史记录
        const userMessage = { type: 'user', content: message };
        setChatHistory(prev => [...prev, userMessage]);

        // 先添加一个空的AI消息用于流式填充
        const aiIndex = chatHistory.length + 1;
        setChatHistory(prev => [...prev, { type: 'assistant', content: '' }]);

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
            const response = await fetch('http://127.0.0.1:5000/chat_stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            });
            if (!response.body) throw new Error('无流式响应体');
            const reader = response.body.getReader();
            let aiContent = '';
            let done = false;
            while (!done) {
                const { value, done: doneReading } = await reader.read();
                done = doneReading;
                if (value) {
                    const chunk = new TextDecoder().decode(value);
                    aiContent += chunk;
                    setChatHistory(prev => {
                        // 只更新最后一条AI消息
                        const updated = [...prev];
                        updated[aiIndex] = { type: 'assistant', content: aiContent };
                        return updated;
                    });
                }
            }
            
            // 播放AI回复语音
            if (speechEnabled && aiContent.trim()) {
                speak(aiContent);
            }
            
            setMessage('');
        } catch (error) {
            setError('与大语言模型交互请求失败，请检查控制台日志');
        } finally {
            setIsLoading(false);
            setIsStreaming(false);
        }
    };

    // 获取语音描述
    const getVoiceDescription = (voice) => {
        const lang = voice.lang || '';
        const isChinese = lang.includes('zh') || lang.includes('cmn');
        const gender = voice.name.toLowerCase().includes('female') || 
                      voice.name.toLowerCase().includes('xiaoxiao') || 
                      voice.name.toLowerCase().includes('xiaoyi') ? '女声' : '男声';
        
        return `${voice.name} (${isChinese ? '中文' : '英文'}, ${gender})`;
    };

    // 推荐语音列表
    const recommendedVoices = [
        'Microsoft Yunxi - Chinese (Simplified)',
        'Microsoft Yunyang - Chinese (Simplified)',
        'Microsoft XiaoxiaoNeural - Chinese (Simplified)',
        'Microsoft XiaoyiNeural - Chinese (Simplified)',
        'Microsoft YunxiNeural - Chinese (Simplified)',
        'Google 普通话（中国大陆）',
        'Google 粤语（香港）',
        'Samantha',
        'Alex',
        'Victoria'
    ];

    // 摄像头相关函数
    const startCamera = async () => {
        try {
            console.log('开始请求摄像头权限...');
            
            // 先检查浏览器支持
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('浏览器不支持getUserMedia API');
            }
            
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 640 }, 
                    height: { ideal: 480 },
                    facingMode: 'user' // 使用前置摄像头
                } 
            });
            
            console.log('摄像头权限获取成功:', stream);
            console.log('视频轨道:', stream.getVideoTracks());
            
            setCameraStream(stream);
            setIsCameraOn(true);
            setError('');
            
            // 等待视频元素准备好
            setTimeout(() => {
                if (videoRef) {
                    console.log('设置视频源...');
                    videoRef.srcObject = stream;
                    
                    // 强制播放
                    videoRef.play().then(() => {
                        console.log('视频开始播放');
                    }).catch(error => {
                        console.error('视频播放失败:', error);
                        setError('视频播放失败: ' + error.message);
                    });
                }
            }, 100);
            
        } catch (error) {
            console.error('摄像头启动失败:', error);
            let errorMessage = '摄像头启动失败';
            
            if (error.name === 'NotAllowedError') {
                errorMessage = '摄像头权限被拒绝，请允许浏览器访问摄像头';
            } else if (error.name === 'NotFoundError') {
                errorMessage = '未找到摄像头设备';
            } else if (error.name === 'NotReadableError') {
                errorMessage = '摄像头被其他应用占用';
            } else if (error.name === 'OverconstrainedError') {
                errorMessage = '摄像头不支持请求的分辨率';
            } else {
                errorMessage = '摄像头启动失败: ' + error.message;
            }
            
            setError(errorMessage);
        }
    };

    const stopCamera = () => {
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            setCameraStream(null);
        }
        setIsCameraOn(false);
    };

    const captureImage = () => {
        if (!videoRef || !canvasRef || !isCameraOn) {
            console.log('摄像头未就绪，无法捕获图像');
            return;
        }
        
        const video = videoRef;
        const canvas = canvasRef;
        
        // 检查视频是否就绪
        if (video.readyState !== video.HAVE_ENOUGH_DATA) {
            console.log('视频数据不足，等待下一帧');
            return;
        }
        
        // 检查视频尺寸
        if (video.videoWidth === 0 || video.videoHeight === 0) {
            console.log('视频尺寸无效，等待视频加载');
            return;
        }
        
        const context = canvas.getContext('2d');
        
        // 设置canvas尺寸与视频相同
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        try {
            // 绘制视频帧到canvas
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            // 将canvas转换为base64图片
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            setImage(imageData);
            
            // 自动检测表情
            handleDetectFaceFromImage(imageData);
        } catch (error) {
            console.error('图像捕获失败:', error);
            setError('图像捕获失败，请重试');
        }
    };

    const handleDetectFaceFromImage = async (imageData) => {
        if (!imageData || isLoading) return;
        setIsLoading(true);
        setError('');
        try {
            const response = await axios.post('http://127.0.0.1:5000/detect_face', { image: imageData });
            setEmotionResult(response.data.result);
            setCurrentEmotion(response.data.result);
        } catch (error) {
            console.error('表情检测失败:', error);
            setError('表情检测失败，请重试');
        } finally {
            setIsLoading(false);
        }
    };

    const clearChat = () => {
        setChatHistory([]);
        setMessage('');
        setError('');
        setSessionStarted(false);
        setSessionId(null);
        setCurrentEmotion('');
        setShowSessionSetup(true);
    };

    return (
        <div className="app-container">
            <div className="sidebar">
                <div className="sidebar-header">
                    <h1 className="app-title">AI情感治疗师</h1>
                </div>
                
                {/* 语音设置面板 */}
                {showVoiceSettings && (
                    <div className="voice-settings-panel">
                        <h3>语音设置</h3>
                        
                        {/* TTS服务选择 */}
                        <div className="setting-group">
                            <label>TTS服务:</label>
                            <select 
                                value={ttsService} 
                                onChange={(e) => setTtsService(e.target.value)}
                            >
                                <option value="browser">浏览器内置语音</option>
                                {ttsServices.azure?.enabled && (
                                    <option value="azure">Azure 语音服务</option>
                                )}
                                {ttsServices.aliyun?.enabled && (
                                    <option value="aliyun">阿里云语音服务</option>
                                )}
                                {ttsServices.baidu?.enabled && (
                                    <option value="baidu">百度语音服务</option>
                                )}
                            </select>
                        </div>

                        {/* 高级TTS服务配置 */}
                        {ttsService !== 'browser' && (
                            <div className="setting-group">
                                <label>语音选择:</label>
                                <select 
                                    value={selectedTtsVoice} 
                                    onChange={(e) => setSelectedTtsVoice(e.target.value)}
                                >
                                    <option value="">请选择语音</option>
                                    {ttsServices[ttsService]?.voices && 
                                        Object.entries(ttsServices[ttsService].voices).map(([key, name]) => (
                                            <option key={key} value={key}>{name}</option>
                                        ))
                                    }
                                </select>
                            </div>
                        )}

                        {/* 浏览器语音设置 */}
                        {ttsService === 'browser' && (
                            <>
                                <div className="setting-group">
                                    <label>语音选择:</label>
                                    <select 
                                        value={selectedVoice ? selectedVoice.name : ''} 
                                        onChange={(e) => {
                                            const voice = availableVoices.find(v => v.name === e.target.value);
                                            setSelectedVoice(voice);
                                        }}
                                    >
                                        <option value="">请选择语音</option>
                                        {availableVoices.map((voice, index) => (
                                            <option key={index} value={voice.name}>
                                                {getVoiceDescription(voice)}
                                            </option>
                                        ))}
                                    </select>
                                </div>

                                <div className="setting-group">
                                    <label>语速: {speechRate.toFixed(1)}</label>
                                    <input 
                                        type="range" 
                                        min="0.5" 
                                        max="2" 
                                        step="0.1" 
                                        value={speechRate} 
                                        onChange={(e) => setSpeechRate(parseFloat(e.target.value))}
                                    />
                                </div>

                                <div className="setting-group">
                                    <label>音调: {speechPitch.toFixed(1)}</label>
                                    <input 
                                        type="range" 
                                        min="0.5" 
                                        max="2" 
                                        step="0.1" 
                                        value={speechPitch} 
                                        onChange={(e) => setSpeechPitch(parseFloat(e.target.value))}
                                    />
                                </div>

                                <div className="setting-group">
                                    <label>音量: {speechVolume.toFixed(1)}</label>
                                    <input 
                                        type="range" 
                                        min="0" 
                                        max="1" 
                                        step="0.1" 
                                        value={speechVolume} 
                                        onChange={(e) => setSpeechVolume(parseFloat(e.target.value))}
                                    />
                                </div>

                                <div className="setting-group">
                                    <label>
                                        <input 
                                            type="checkbox" 
                                            checked={useAdvancedVoice} 
                                            onChange={(e) => setUseAdvancedVoice(e.target.checked)}
                                        />
                                        使用高级语音优化
                                    </label>
                                </div>

                                {useAdvancedVoice && (
                                    <>
                                        <div className="setting-group">
                                            <label>语音风格:</label>
                                            <select 
                                                value={voiceStyle} 
                                                onChange={(e) => setVoiceStyle(e.target.value)}
                                            >
                                                <option value="natural">自然</option>
                                                <option value="warm">温暖</option>
                                                <option value="professional">专业</option>
                                            </select>
                                        </div>

                                        <div className="setting-group">
                                            <label>
                                                <input 
                                                    type="checkbox" 
                                                    checked={pauseBetweenSentences} 
                                                    onChange={(e) => setPauseBetweenSentences(e.target.checked)}
                                                />
                                                句子间添加停顿
                                            </label>
                                        </div>
                                    </>
                                )}
                            </>
                        )}

                        {/* 推荐语音 */}
                        {ttsService === 'browser' && (
                            <div className="setting-group">
                                <label>推荐语音:</label>
                                <div className="recommended-voices">
                                    {recommendedVoices.map((voiceName, index) => {
                                        const voice = availableVoices.find(v => 
                                            v.name.toLowerCase().includes(voiceName.toLowerCase().split(' ')[0])
                                        );
                                        return voice ? (
                                            <button 
                                                key={index}
                                                className="voice-button"
                                                onClick={() => setSelectedVoice(voice)}
                                            >
                                                {getVoiceDescription(voice)}
                                            </button>
                                        ) : null;
                                    })}
                                </div>
                            </div>
                        )}

                        {/* 语音测试 */}
                        <div className="setting-group">
                            <button 
                                className="test-voice-btn"
                                onClick={() => speak("你好，我是你的AI治疗师。让我们开始今天的对话吧。")}
                                disabled={isSpeaking}
                            >
                                {isSpeaking ? '播放中...' : '测试语音'}
                            </button>
                            {isSpeaking && (
                                <button 
                                    className="stop-voice-btn"
                                    onClick={stopSpeaking}
                                >
                                    停止
                                </button>
                            )}
                        </div>

                        <div className="setting-group">
                            <label>
                                <input 
                                    type="checkbox" 
                                    checked={speechEnabled} 
                                    onChange={(e) => setSpeechEnabled(e.target.checked)}
                                />
                                启用语音播放
                            </label>
                        </div>

                        <div className="setting-group">
                            <label>
                                <input 
                                    type="checkbox" 
                                    checked={voiceEnabled} 
                                    onChange={(e) => setVoiceEnabled(e.target.checked)}
                                />
                                启用语音输入
                            </label>
                        </div>

                        <button 
                            className="close-settings-btn"
                            onClick={() => setShowVoiceSettings(false)}
                        >
                            关闭设置
                        </button>
                    </div>
                )}
                
                {/* 会话设置区域 */}
                {showSessionSetup && (
                    <div className="session-setup">
                        <h2>开始治疗会话</h2>
                        <div className="therapy-type-selector">
                            <label htmlFor="therapy-type">选择治疗模式</label>
                            <select
                                id="therapy-type"
                                value={therapyType}
                                onChange={(e) => setTherapyType(e.target.value)}
                                className="model-select"
                            >
                                <option value="default">一般心理咨询</option>
                                <option value="cbt">认知行为治疗(CBT)</option>
                                <option value="mindfulness">正念治疗</option>
                            </select>
                        </div>
                        
                        <div className="model-selector">
                            <label htmlFor="model-select">选择AI模型</label>
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
                        
                        <button 
                            onClick={startTherapySession} 
                            disabled={isLoading}
                            className={isLoading ? 'btn loading' : 'btn primary'}
                        >
                            {isLoading ? '启动中...' : '开始治疗会话'}
                        </button>
                    </div>
                )}
                
                {/* 会话控制区域 */}
                {sessionStarted && (
                    <div className="session-controls">
                        <div className="session-info">
                            <h3>当前会话</h3>
                            <p>治疗模式: {therapyType === 'default' ? '一般心理咨询' : 
                                         therapyType === 'cbt' ? '认知行为治疗' : '正念治疗'}</p>
                            <p>当前情绪: {currentEmotion || '未检测'}</p>
                            <p>对话轮数: {Math.floor(chatHistory.length / 2)}</p>
                        </div>
                        <button 
                            onClick={endTherapySession} 
                            disabled={isLoading}
                            className="btn secondary"
                        >
                            结束会话
                        </button>
                    </div>
                )}
                
                <div className="emotion-section">
                    <h2>表情识别</h2>
                    
                    {/* 摄像头控制区域 */}
                    <div className="camera-controls">
                        <div className="camera-buttons">
                            {!isCameraOn ? (
                                <button 
                                    onClick={startCamera}
                                    className="btn primary"
                                    disabled={isLoading}
                                >
                                    📷 开启摄像头
                                </button>
                            ) : (
                                <>
                                    <button 
                                        onClick={stopCamera}
                                        className="btn secondary"
                                    >
                                        🛑 关闭摄像头
                                    </button>
                                    
                                    <button 
                                        onClick={captureImage}
                                        className="btn primary"
                                        disabled={isLoading}
                                    >
                                        📸 拍照
                                    </button>
                                    
                                    <button 
                                        onClick={() => {
                                            console.log('视频状态:', videoRef?.readyState);
                                            console.log('视频尺寸:', videoRef?.videoWidth, 'x', videoRef?.videoHeight);
                                            console.log('摄像头流:', cameraStream);
                                        }}
                                        className="btn secondary"
                                    >
                                        🔍 调试信息
                                    </button>
                                    
                                    <button 
                                        onClick={() => {
                                            if (videoRef && cameraStream) {
                                                console.log('强制刷新视频...');
                                                videoRef.srcObject = null;
                                                setTimeout(() => {
                                                    videoRef.srcObject = cameraStream;
                                                    videoRef.play().catch(console.error);
                                                }, 100);
                                            }
                                        }}
                                        className="btn secondary"
                                    >
                                        🔄 刷新视频
                                    </button>
                                </>
                            )}
                        </div>
                    </div>
                    
                    {/* 摄像头预览 */}
                    {isCameraOn && (
                        <div className="camera-preview">
                            <video
                                ref={setVideoRef}
                                autoPlay
                                playsInline
                                muted
                                controls={false}
                                style={{ 
                                    width: '100%', 
                                    maxWidth: '400px', 
                                    height: '300px',
                                    backgroundColor: '#000',
                                    border: '2px solid #ccc',
                                    borderRadius: '8px'
                                }}
                            />
                            <canvas
                                ref={setCanvasRef}
                                style={{ display: 'none' }}
                            />
                            
                            {/* 摄像头状态信息 */}
                            <div className="camera-status">
                                <p>摄像头状态: {videoRef && videoRef.readyState === videoRef.HAVE_ENOUGH_DATA ? '✅ 就绪' : '⏳ 加载中...'}</p>
                                {videoRef && (
                                    <p>视频尺寸: {videoRef.videoWidth} x {videoRef.videoHeight}</p>
                                )}
                                <p>流状态: {cameraStream ? '✅ 已连接' : '❌ 未连接'}</p>
                            </div>
                        </div>
                    )}
                    
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
                    <div className="chat-header">
                        {sessionStarted ? (
                            <h2>治疗对话 - {therapyType === 'default' ? '一般心理咨询' : 
                                           therapyType === 'cbt' ? '认知行为治疗' : '正念治疗'}</h2>
                        ) : (
                            <h2>智能对话</h2>
                        )}
                    </div>
                    
                    <div className="chat-messages">
                        {chatHistory.length === 0 ? (
                            <div className="empty-chat">
                                <p>开始你的治疗之旅吧！</p>
                                <p>上传照片检测表情，然后开始与AI治疗师对话。</p>
                            </div>
                        ) : (
                            chatHistory.map((msg, index) => (
                                <div key={index} className={`message ${msg.type}`}>
                                    <div className="message-content">
                                        {msg.type === 'assistant' && (
                                            <div className="assistant-avatar">治疗师</div>
                                        )}
                                        <div className="message-text">
                                            {msg.content}
                                            {/* 打字机光标，仅在最后一条AI消息流式时显示 */}
                                            {msg.type === 'assistant' && index === chatHistory.length - 1 && isStreaming && (
                                                <span className="typing-cursor"></span>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            ))
                        )}
                        <div ref={messagesEndRef} />
                    </div>
                    
                    <div className="chat-input-wrapper">
                        <div className="chat-input-container">
                            <textarea
                                ref={textareaRef}
                                value={message}
                                onChange={(e) => setMessage(e.target.value)}
                                onKeyPress={handleKeyPress}
                                placeholder={sessionStarted ? "与治疗师分享你的感受..." : "输入消息与智能助手对话..."}
                                disabled={isLoading}
                                className="chat-input"
                                rows="1"
                                style={{resize: 'none', overflow: 'hidden'}}
                            />
                            <div className="chat-input-actions">
                                {/* 语音输入按钮 */}
                                {voiceEnabled && (
                                    <button
                                        onClick={isListening ? stopListening : startListening}
                                        className={`voice-button ${isListening ? 'listening' : ''}`}
                                        disabled={isLoading}
                                        title={isListening ? '停止录音' : '开始录音'}
                                    >
                                        {isListening ? '🔴' : '🎤'}
                                    </button>
                                )}
                                
                                {/* 语音播放控制 */}
                                {speechEnabled && (
                                    <button
                                        onClick={isSpeaking ? stopSpeaking : () => {
                                            const lastMessage = chatHistory[chatHistory.length - 1];
                                            if (lastMessage && lastMessage.type === 'assistant') {
                                                speak(lastMessage.content);
                                            }
                                        }}
                                        className={`voice-button ${isSpeaking ? 'speaking' : ''}`}
                                        disabled={isLoading}
                                        title={isSpeaking ? '停止播放' : '重新播放'}
                                    >
                                        {isSpeaking ? '⏹️' : '🔊'}
                                    </button>
                                )}
                                
                                <button 
                                    onClick={sessionStarted ? handleTherapyChat : handleChat} 
                                    disabled={isLoading || !message.trim()}
                                    className={isLoading ? 'send-button loading' : 'send-button'}
                                >
                                    {isLoading ? '' : '发送'}
                                </button>
                            </div>
                        </div>
                        
                        {/* 语音识别状态显示 */}
                        {isListening && (
                            <div className="voice-status">
                                <p>正在听您说话... {transcript && `"${transcript}"`}</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* 控制按钮区域 */}
            <div className="control-buttons">
                <button 
                    className={`btn ${isListening ? 'listening' : ''}`}
                    onClick={isListening ? stopListening : startListening}
                    disabled={!voiceEnabled}
                >
                    {isListening ? '🎤 停止录音' : '🎤 语音输入'}
                </button>
                
                <button 
                    className="btn secondary"
                    onClick={() => setShowVoiceSettings(!showVoiceSettings)}
                >
                    ⚙️ 语音设置
                </button>
                
                <button 
                    className="btn secondary"
                    onClick={clearChat}
                >
                    🗑️ 清空对话
                </button>
            </div>

            {error && <div className="error-message">{error}</div>}
        </div>
    );
}

export default App;