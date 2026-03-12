import React, { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality } from '@google/genai';
import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import { TargetLanguage, ChatMessage, OfflinePack, ArchivedSession, AIProvider, APIKeys } from './types';
import { float32To16BitPCM, arrayBufferToBase64, base64ToArrayBuffer, pcm16ToFloat32 } from './utils/audioUtils';
import AudioVisualizer from './components/AudioVisualizer';
import { Mic, Globe, Settings, RotateCcw, Wifi, WifiOff, Download, Check, Trash2, X, Zap, Square, Send, ChevronDown, Sparkles, Loader2, Languages, ArrowRightLeft, ArrowRight, User, SplitSquareVertical, Maximize2, Minimize2, MessageSquare, Ear, ScrollText, Save, FolderOpen, Calendar, ChevronRight, FileText, Waves, Key, LogOut, ExternalLink, Keyboard, History, BookOpen, Volume2, Camera, RefreshCw } from 'lucide-react';

// Live API Configuration
const MODEL_NAME = 'gemini-2.5-flash-native-audio-preview-09-2025';
const INPUT_SAMPLE_RATE = 16000;
const OUTPUT_SAMPLE_RATE = 24000;

// Language Metadata
const LANGUAGE_META = [
  { code: TargetLanguage.TURKISH, name: 'Türkçe', flag: '🇹🇷', short: 'TR', locale: 'tr-TR' },
  { code: TargetLanguage.ENGLISH, name: 'İngilizce', flag: '🇬🇧', short: 'EN', locale: 'en-US' },
  { code: TargetLanguage.DUTCH, name: 'Hollandaca', flag: '🇳🇱', short: 'NL', locale: 'nl-NL' },
  { code: TargetLanguage.ARABIC, name: 'Arapça', flag: '🇸🇦', short: 'AR', locale: 'ar-SA' },
  { code: TargetLanguage.RUSSIAN, name: 'Rusça', flag: '🇷🇺', short: 'RU', locale: 'ru-RU' },
  { code: TargetLanguage.GERMAN, name: 'Almanca', flag: '🇩🇪', short: 'DE', locale: 'de-DE' },
  { code: TargetLanguage.ITALIAN, name: 'İtalyanca', flag: '🇮🇹', short: 'IT', locale: 'it-IT' },
  { code: TargetLanguage.FRENCH, name: 'Fransızca', flag: '🇫🇷', short: 'FR', locale: 'fr-FR' },
  { code: TargetLanguage.UKRAINIAN, name: 'Ukraynaca', flag: '🇺🇦', short: 'UA', locale: 'uk-UA' },
  { code: TargetLanguage.PORTUGUESE, name: 'Portekizce', flag: '🇵🇹', short: 'PT', locale: 'pt-PT' },
  { code: TargetLanguage.CHINESE, name: 'Çince', flag: '🇨🇳', short: 'ZH', locale: 'zh-CN' },
  { code: TargetLanguage.SPANISH, name: 'İspanyolca', flag: '🇪🇸', short: 'ES', locale: 'es-ES' },
  { code: TargetLanguage.JAPANESE, name: 'Japonca', flag: '🇯🇵', short: 'JA', locale: 'ja-JP' },
];

const getLangDetails = (lang?: TargetLanguage) => {
  if (!lang) return LANGUAGE_META[0];
  return LANGUAGE_META.find(l => l.code === lang) || LANGUAGE_META[0];
};

// Advanced Client-Side Language Detection
const COMMON_WORDS: Record<string, Set<string>> = {
  [TargetLanguage.TURKISH]: new Set(['ve', 'bir', 'bu', 'da', 'de', 'için', 'ben', 'sen', 'o', 'biz', 'siz', 'evet', 'hayır']),
  [TargetLanguage.ENGLISH]: new Set(['the', 'and', 'is', 'it', 'to', 'in', 'you', 'that', 'of', 'for', 'on', 'are', 'hello']),
  [TargetLanguage.ARABIC]: new Set(['من', 'في', 'على', 'ان', 'لا', 'ما', 'هو', 'يا', 'نحن', 'هذا']),
};

const detectLanguage = (text: string, langA: TargetLanguage, langB: TargetLanguage): TargetLanguage => {
  if (!text || text.length < 2) return langA;
  const t = text.toLowerCase().trim();

  // Script checks
  if (/[\u0600-\u06FF]/.test(text)) return TargetLanguage.ARABIC;
  if (/[\u0400-\u04FF]/.test(text)) {
    return (text.includes('і') || text.includes('є')) ? TargetLanguage.UKRAINIAN : TargetLanguage.RUSSIAN;
  }
  if (/[ğıİşŞ]/.test(text)) return TargetLanguage.TURKISH;

  const getScore = (l: TargetLanguage) => {
    if (!COMMON_WORDS[l]) return 0;
    return t.split(/\s+/).filter(w => COMMON_WORDS[l].has(w)).length;
  };

  const scoreA = getScore(langA);
  const scoreB = getScore(langB);

  if (scoreA > scoreB) return langA;
  if (scoreB > scoreA) return langB;
  return langA;
};

const INITIAL_PACKS: OfflinePack[] = [
  { id: 'tr-en', pair: 'TR ↔ EN', name: 'İngilizce', size: '45 MB', downloaded: false, progress: 0 },
  { id: 'tr-nl', pair: 'TR ↔ NL', name: 'Hollandaca', size: '42 MB', downloaded: false, progress: 0 },
  { id: 'tr-de', pair: 'TR ↔ DE', name: 'Almanca', size: '44 MB', downloaded: false, progress: 0 },
];

const App: React.FC = () => {
  const [sourceLang, setSourceLang] = useState<TargetLanguage>(TargetLanguage.TURKISH);
  const [targetLang, setTargetLang] = useState<TargetLanguage>(TargetLanguage.ENGLISH);
  const [apiKeys, setApiKeys] = useState<APIKeys>({});
  const [selectedProvider, setSelectedProvider] = useState<AIProvider>(AIProvider.GEMINI);
  const [tempApiKeyInput, setTempApiKeyInput] = useState('');
  const [tempProviderInput, setTempProviderInput] = useState<AIProvider>(AIProvider.GEMINI);
  const [viewMode, setViewMode] = useState<'chat' | 'split' | 'listen' | 'archive' | 'photo'>('chat');
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<{ translation: string, info: string } | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // Voice Preference: 'female' | 'male'
  const [voiceType, setVoiceType] = useState<'female' | 'male'>(() => {
    const saved = localStorage.getItem('voiceType');
    return (saved === 'male' || saved === 'female') ? saved : 'female';
  });

  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [realtimeInput, setRealtimeInput] = useState('');
  const [realtimeOutput, setRealtimeOutput] = useState('');
  const [isNoiseMode, setIsNoiseMode] = useState(false);
  const [isHoldingMic, setIsHoldingMic] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [isListenModeActive, setIsListenModeActive] = useState(false);
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [savedSessions, setSavedSessions] = useState<ArchivedSession[]>([]);
  const [openedArchive, setOpenedArchive] = useState<ArchivedSession | null>(null);
  const [offlinePacks, setOfflinePacks] = useState<OfflinePack[]>(INITIAL_PACKS);
  const [showLangSelector, setShowLangSelector] = useState<'source' | 'target' | null>(null);
  const [textInput, setTextInput] = useState('');
  const [isTextTranslating, setIsTextTranslating] = useState(false);
  const [isKeyboardVisible, setIsKeyboardVisible] = useState(false);
  const [isSpeechPlaying, setIsSpeechPlaying] = useState<string | null>(null);
  const [showUpdates, setShowUpdates] = useState(false);
  const [showGuide, setShowGuide] = useState(false);

  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const inputAnalyserRef = useRef<AnalyserNode | null>(null);
  const outputAnalyserRef = useRef<AnalyserNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const aiClientRef = useRef<GoogleGenAI | null>(null);
  const sessionPromiseRef = useRef<Promise<any> | null>(null);
  const activeSessionRef = useRef<any>(null);
  const nextStartTimeRef = useRef<number>(0);
  const currentInputTranscription = useRef('');
  const currentOutputTranscription = useRef('');

  useEffect(() => {
    const storedKeys = localStorage.getItem('ai_api_keys');
    const storedProvider = localStorage.getItem('selected_provider');
    
    if (storedKeys) {
      const keys = JSON.parse(storedKeys);
      setApiKeys(keys);
      if (keys.gemini) {
        aiClientRef.current = new GoogleGenAI({ apiKey: keys.gemini });
      }
    } else {
      // Fallback for old single key storage
      const oldKey = localStorage.getItem('gemini_api_key');
      if (oldKey) {
        const keys = { gemini: oldKey };
        setApiKeys(keys);
        localStorage.setItem('ai_api_keys', JSON.stringify(keys));
        aiClientRef.current = new GoogleGenAI({ apiKey: oldKey });
      } else {
        const envKey = (typeof process !== 'undefined' && process.env) ? process.env.API_KEY : undefined;
        if (envKey) {
          const keys = { gemini: envKey };
          setApiKeys(keys);
          aiClientRef.current = new GoogleGenAI({ apiKey: envKey });
        }
      }
    }

    if (storedProvider) {
      setSelectedProvider(storedProvider as AIProvider);
    }

    const savedArchive = localStorage.getItem('archivedSessions');
    if (savedArchive) setSavedSessions(JSON.parse(savedArchive));
  }, []);

  useEffect(() => {
    if (apiKeys.gemini) {
      aiClientRef.current = new GoogleGenAI({ apiKey: apiKeys.gemini });
    } else {
      const envKey = (typeof process !== 'undefined' ? (process.env.GEMINI_API_KEY || process.env.API_KEY) : undefined);
      if (envKey) {
        aiClientRef.current = new GoogleGenAI({ apiKey: envKey });
      }
    }
  }, [apiKeys.gemini]);

  const triggerHaptic = () => { if (navigator.vibrate) navigator.vibrate(50); };

  const handleVoiceChange = (type: 'female' | 'male') => {
    setVoiceType(type);
    localStorage.setItem('voiceType', type);
    triggerHaptic();
    
    // If connected online, reconnect to apply new voice
    if (isConnected && !isListenModeActive) {
      stopConnection();
      setTimeout(() => startLiveSession('bidirectional'), 300);
    }
  };

  useEffect(() => {
    if (videoRef.current && cameraStream && viewMode === 'photo') {
      videoRef.current.srcObject = cameraStream;
    }
  }, [cameraStream, viewMode]);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment' } 
      });
      setCameraStream(stream);
      setViewMode('photo');
      triggerHaptic();
    } catch (err) {
      console.error("Camera error:", err);
      setError("Kamera erişimi sağlanamadı.");
    }
  };

  const stopCamera = () => {
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      setCameraStream(null);
    }
    setCapturedImage(null);
    setAnalysisResult(null);
  };

  const handleCapture = async () => {
    if (!videoRef.current || !canvasRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const imageData = canvas.toDataURL('image/jpeg');
      setCapturedImage(imageData);
      analyzeImage(imageData);
      triggerHaptic();
    }
  };

  const analyzeImage = async (base64Image: string) => {
    setIsAnalyzing(true);
    setAnalysisResult(null);
    
    const sourceDetails = getLangDetails(sourceLang);
    const targetDetails = getLangDetails(targetLang);

    try {
      const apiKey = apiKeys.gemini || (typeof process !== 'undefined' ? (process.env.GEMINI_API_KEY || process.env.API_KEY) : '');
      if (!apiKey) throw new Error("API Anahtarı bulunamadı.");

      const ai = new GoogleGenAI({ apiKey });
      const base64Data = base64Image.split(',')[1];
      
      const response = await ai.models.generateContent({
        model: 'gemini-3-flash-preview',
        contents: {
          parts: [
            { text: `Bu görseldeki ürünü analiz et. 
              1. Görseldeki metinleri veya ürün ismini "${targetDetails.name}" diline çevir.
              2. Ürün hakkında "${targetDetails.name}" dilinde çok kısa ve öz (maksimum 2 cümle) bilgi ver.
              Yanıtı şu JSON formatında ver: {"translation": "çeviri", "info": "bilgi"}` 
            },
            { inlineData: { mimeType: 'image/jpeg', data: base64Data } }
          ]
        },
        config: {
          responseMimeType: 'application/json'
        }
      });

      const result = JSON.parse(response.text);
      setAnalysisResult(result);
    } catch (err: any) {
      console.error("Analysis error:", err);
      setError("Görsel analiz edilemedi.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const stopConnection = useCallback(() => {
    setIsConnecting(false); setIsConnected(false); setIsListenModeActive(false);
    setRealtimeInput(''); setRealtimeOutput('');
    if (mediaStreamRef.current) mediaStreamRef.current.getTracks().forEach(t => t.stop());
    if (scriptProcessorRef.current) scriptProcessorRef.current.disconnect();
    if (inputAudioContextRef.current && inputAudioContextRef.current.state !== 'closed') inputAudioContextRef.current.close();
    if (outputAudioContextRef.current && outputAudioContextRef.current.state !== 'closed') outputAudioContextRef.current.close();
    if (activeSessionRef.current) activeSessionRef.current.close();
    sessionPromiseRef.current = null;
    triggerHaptic();
  }, []);

  const handleTextSubmit = async () => {
    if (!textInput.trim() || isTextTranslating) return;
    
    const textToTranslate = textInput.trim();
    setIsTextTranslating(true);
    triggerHaptic();

    const sourceDetails = getLangDetails(sourceLang);
    const targetDetails = getLangDetails(targetLang);

    try {
      let translatedText = '';

      if (selectedProvider === AIProvider.GEMINI) {
        const apiKey = apiKeys.gemini || (typeof process !== 'undefined' ? (process.env.GEMINI_API_KEY || process.env.API_KEY) : '');
        if (!apiKey) throw new Error("Gemini API anahtarı bulunamadı. Lütfen ayarlardan girin.");
        
        const ai = new GoogleGenAI({ apiKey });
        const response = await ai.models.generateContent({
          model: 'gemini-3-flash-preview',
          contents: textToTranslate,
          config: {
            systemInstruction: `Sen bir tercümansın. GÖREVİN: "${sourceDetails.name}" dilindeki metni "${targetDetails.name}" diline çevirmek. SADECE çeviriyi döndür, başka açıklama yapma.`,
          }
        });
        translatedText = response.text || '';
      } else if (selectedProvider === AIProvider.OPENAI) {
        if (!apiKeys.openai) throw new Error("OpenAI API anahtarı bulunamadı. Lütfen ayarlardan girin.");
        const openai = new OpenAI({ apiKey: apiKeys.openai, dangerouslyAllowBrowser: true });
        const response = await openai.chat.completions.create({
          model: 'gpt-4o',
          messages: [
            { role: 'system', content: `Sen bir tercümansın. GÖREVİN: "${sourceDetails.name}" dilindeki metni "${targetDetails.name}" diline çevirmek. SADECE çeviriyi döndür, başka açıklama yapma.` },
            { role: 'user', content: textToTranslate }
          ]
        });
        translatedText = response.choices[0].message.content || '';
      } else if (selectedProvider === AIProvider.ANTHROPIC) {
        if (!apiKeys.anthropic) throw new Error("Anthropic API anahtarı bulunamadı. Lütfen ayarlardan girin.");
        const anthropic = new Anthropic({ apiKey: apiKeys.anthropic, dangerouslyAllowBrowser: true });
        const response = await anthropic.messages.create({
          model: 'claude-3-5-sonnet-20240620',
          max_tokens: 1024,
          system: `Sen bir tercümansın. GÖREVİN: "${sourceDetails.name}" dilindeki metni "${targetDetails.name}" diline çevirmek. SADECE çeviriyi döndür, başka açıklama yapma.`,
          messages: [{ role: 'user', content: textToTranslate }]
        });
        translatedText = (response.content[0] as any).text || '';
      }
      
      if (!translatedText) throw new Error("Çeviri alınamadı.");
      
      setMessages(prev => [...prev, 
        { id: Date.now().toString(), role: 'user', text: textToTranslate, timestamp: new Date(), isFinal: true },
        { id: (Date.now() + 1).toString(), role: 'model', text: translatedText, timestamp: new Date(), isFinal: true, langCode: targetLang }
      ]);
      setTextInput(''); // Only clear on success
    } catch (error: any) {
      console.error("Text translation error:", error);
      setError(error.message || "Çeviri sırasında bir hata oluştu.");
    } finally {
      setIsTextTranslating(false);
    }
  };

  const handlePlaySpeech = async (text: string, messageId: string) => {
    if (!text || isSpeechPlaying) return;
    
    setIsSpeechPlaying(messageId);
    triggerHaptic();

    try {
      const apiKey = apiKeys.gemini || (typeof process !== 'undefined' ? (process.env.GEMINI_API_KEY || process.env.API_KEY) : '');
      if (!apiKey) throw new Error("API Anahtarı bulunamadı.");

      const ai = new GoogleGenAI({ apiKey });
      const response = await ai.models.generateContent({
        model: "gemini-2.5-flash-preview-tts",
        contents: [{ parts: [{ text }] }],
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName: voiceType === 'female' ? 'Kore' : 'Fenrir' },
            },
          },
        },
      });

      const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
      if (base64Audio) {
        const arrayBuffer = base64ToArrayBuffer(base64Audio);
        const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
        const float32Data = pcm16ToFloat32(arrayBuffer);
        const buffer = audioCtx.createBuffer(1, float32Data.length, 24000);
        buffer.getChannelData(0).set(float32Data);
        const source = audioCtx.createBufferSource();
        source.buffer = buffer;
        source.connect(audioCtx.destination);
        source.onended = () => {
          setIsSpeechPlaying(null);
          audioCtx.close();
        };
        source.start();
      } else {
        throw new Error("Ses verisi alınamadı.");
      }
    } catch (err: any) {
      console.error("TTS Error:", err);
      setError(err.message || "Ses çalınırken bir hata oluştu.");
      setIsSpeechPlaying(null);
    }
  };

  const startLiveSession = async (mode: 'bidirectional' | 'listen') => {
    if (isConnecting || isConnected) {
      if (isConnected) {
        if (mode === 'listen' && messages.length > 0) setShowSaveModal(true);
        else stopConnection();
      }
      return;
    }
    setIsConnecting(true); setError(null);
    const isListen = mode === 'listen';
    if (isListen) { setViewMode('listen'); setIsListenModeActive(true); setMessages([]); }

    try {
      if (!aiClientRef.current) {
        const stored = localStorage.getItem('gemini_api_key');
        if (stored) aiClientRef.current = new GoogleGenAI({ apiKey: stored });
        else throw new Error("API Anahtarı eksik.");
      }

      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: false, // Disabled to prevent filtering out distant voices as noise
          autoGainControl: true,
          channelCount: 1
        } 
      });
      mediaStreamRef.current = stream;
      const audioCtx = new AudioContext({ sampleRate: INPUT_SAMPLE_RATE });
      inputAudioContextRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(stream);
      
      // Sensitivity Boost: Add a GainNode to amplify low signals
      const gainNode = audioCtx.createGain();
      // Boost gain significantly in listen mode to catch distant sounds
      gainNode.gain.value = isListen ? 2.5 : 1.5; 

      // Dynamics Compressor: Helps normalize distant and near voices
      const compressor = audioCtx.createDynamicsCompressor();
      compressor.threshold.setValueAtTime(-50, audioCtx.currentTime);
      compressor.knee.setValueAtTime(40, audioCtx.currentTime);
      compressor.ratio.setValueAtTime(12, audioCtx.currentTime);
      compressor.attack.setValueAtTime(0, audioCtx.currentTime);
      compressor.release.setValueAtTime(0.25, audioCtx.currentTime);

      const analyser = audioCtx.createAnalyser();
      inputAnalyserRef.current = analyser;
      const scriptProcessor = audioCtx.createScriptProcessor(4096, 1, 1);
      scriptProcessorRef.current = scriptProcessor;
      
      scriptProcessor.onaudioprocess = (e) => {
        if (isNoiseMode && !isHoldingMic) return;
        const pcmData = float32To16BitPCM(e.inputBuffer.getChannelData(0));
        sessionPromiseRef.current?.then(s => s.sendRealtimeInput({ 
          media: { mimeType: 'audio/pcm;rate=16000', data: arrayBufferToBase64(pcmData) } 
        }));
      };

      // Audio Chain: Source -> Gain -> Compressor -> Analyser -> ScriptProcessor
      source.connect(gainNode);
      gainNode.connect(compressor);
      compressor.connect(analyser);
      analyser.connect(scriptProcessor);
      scriptProcessor.connect(audioCtx.destination);

      const outCtx = new AudioContext({ sampleRate: OUTPUT_SAMPLE_RATE });
      outputAudioContextRef.current = outCtx;
      const outAnalyser = outCtx.createAnalyser();
      outputAnalyserRef.current = outAnalyser;
      outAnalyser.connect(outCtx.destination);

      const sessionPromise = aiClientRef.current.live.connect({
        model: MODEL_NAME,
        config: {
          systemInstruction: isListen ? 
            `Sen SADECE bir simultane tercümansın. GÖREVİN: ${getLangDetails(targetLang).name} dilinde duyduğun her şeyi ANINDA ve BİREBİR ${getLangDetails(sourceLang).name} diline çevirmek. 
             KESİNLİKLE kendi yorumunu katma, sorulara cevap verme, tavsiye verme veya sohbete girme. 
             Eğer bir soru duyarsan, o soruyu cevaplamak yerine ${getLangDetails(sourceLang).name} diline çevir. 
             Sadece çeviriyi seslendir. Başka hiçbir şey söyleme.` : 
            `Sen SADECE bir simultane tercümansın. GÖREVİN: ${getLangDetails(sourceLang).name} ve ${getLangDetails(targetLang).name} dilleri arasında ANINDA ve BİREBİR çeviri yapmak. 
             KESİNLİKLE kendi yorumunu katma, sorulara cevap verme, tavsiye verme veya sohbete girme. 
             Eğer bir soru duyarsan, o soruyu cevaplamak yerine diğer dile çevir. 
             Sadece çeviriyi seslendir. Başka hiçbir şey söyleme.`,
          responseModalities: [Modality.AUDIO],
          inputAudioTranscription: {}, outputAudioTranscription: {},
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName: voiceType === 'female' ? 'Kore' : 'Fenrir' }
            }
          }
        },
        callbacks: {
          onopen: () => { setIsConnecting(false); setIsConnected(true); triggerHaptic(); },
          onmessage: (msg: LiveServerMessage) => handleServerMessage(msg, isListen),
          onclose: stopConnection,
          onerror: (e) => { console.error(e); setError("Bağlantı Hatası"); stopConnection(); }
        }
      });
      sessionPromiseRef.current = sessionPromise;
      sessionPromise.then(s => activeSessionRef.current = s);
    } catch (e: any) { setError(e.message); stopConnection(); }
  };

  const handleServerMessage = (msg: LiveServerMessage, isListen: boolean) => {
    if (msg.serverContent?.inputTranscription) {
      currentInputTranscription.current += msg.serverContent.inputTranscription.text;
      setRealtimeInput(currentInputTranscription.current);
    }
    if (msg.serverContent?.outputTranscription) {
      currentOutputTranscription.current += msg.serverContent.outputTranscription.text;
      setRealtimeOutput(currentOutputTranscription.current);
    }
    const audio = msg.serverContent?.modelTurn?.parts?.find(p => p.inlineData)?.inlineData?.data;
    if (audio && !isListen) playAudio(audio);
    
    if (msg.serverContent?.turnComplete) {
      const input = currentInputTranscription.current.trim();
      const output = currentOutputTranscription.current.trim();
      if (input || output) {
        setMessages(prev => [...prev, 
          { id: Date.now().toString(), role: 'user', text: input || '...', timestamp: new Date(), isFinal: true },
          { id: Date.now().toString() + 'm', role: 'model', text: output || '...', timestamp: new Date(), isFinal: true, langCode: isListen ? sourceLang : targetLang }
        ]);
      }
      currentInputTranscription.current = ''; currentOutputTranscription.current = '';
      setRealtimeInput(''); setRealtimeOutput('');
    }
  };

  const playAudio = async (base64: string) => {
    const ctx = outputAudioContextRef.current;
    if (!ctx) return;
    const arrayBuffer = base64ToArrayBuffer(base64);
    const float32Data = pcm16ToFloat32(arrayBuffer);
    const buffer = ctx.createBuffer(1, float32Data.length, OUTPUT_SAMPLE_RATE);
    buffer.getChannelData(0).set(float32Data);
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(outputAnalyserRef.current!);
    source.start(Math.max(nextStartTimeRef.current, ctx.currentTime));
    nextStartTimeRef.current = Math.max(nextStartTimeRef.current, ctx.currentTime) + buffer.duration;
  };

  const saveSession = () => {
    const newSession: ArchivedSession = {
      id: Date.now().toString(),
      date: new Date().toISOString(),
      targetLang: targetLang,
      preview: messages.length > 0 ? messages[0].text.substring(0, 50) + "..." : "Boş Oturum",
      messages: [...messages]
    };
    const updated = [newSession, ...savedSessions];
    setSavedSessions(updated);
    localStorage.setItem('archivedSessions', JSON.stringify(updated));
    setShowSaveModal(false);
    stopConnection();
  };

  const deleteArchivedSession = (id: string) => {
    const updated = savedSessions.filter(s => s.id !== id);
    setSavedSessions(updated);
    localStorage.setItem('archivedSessions', JSON.stringify(updated));
    if (openedArchive?.id === id) setOpenedArchive(null);
  };

  const sourceDetails = getLangDetails(sourceLang);
  const targetDetails = getLangDetails(targetLang);

  return (
    <div className={`h-[100dvh] w-full flex flex-col font-sans relative overflow-hidden bg-slate-950 text-slate-100 ${isListenModeActive ? 'bg-orange-950 text-orange-50' : ''}`}>
      
      {/* Background Blobs */}
      <div className="absolute inset-0 z-0 pointer-events-none">
        <div className={`absolute top-[-10%] left-[-20%] w-[150vw] h-[60vh] rounded-full blur-[80px] opacity-20 transition-colors duration-1000 ${isListenModeActive ? 'bg-orange-600' : isConnected ? 'bg-emerald-900' : 'bg-blue-900'}`}></div>
      </div>

      {/* HEADER */}
      <header className="z-20 h-16 flex items-center justify-between px-5 backdrop-blur-md bg-black/40 pt-[env(safe-area-inset-top)]">
        <div className="flex items-center gap-3">
          <div className={`w-8 h-8 rounded-lg flex items-center justify-center shadow-lg ${isListenModeActive ? 'bg-orange-500' : 'bg-blue-600'}`}>
            {isListenModeActive ? <Ear size={18} /> : <Languages size={18} />}
          </div>
          <div className="flex flex-col">
            <span className="font-bold text-lg leading-none">Ai Live Translate</span>
            <span className="text-[10px] text-slate-400 mt-1 uppercase tracking-widest">{isListenModeActive ? 'Dinleme Modu' : 'Developer by Ali TELLIOGLU'}</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button 
            onClick={() => {
              if (viewMode === 'photo') {
                stopCamera();
                setViewMode('chat');
              } else {
                startCamera();
              }
              triggerHaptic();
            }} 
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full transition-all ${viewMode === 'photo' ? 'bg-emerald-600 text-white' : 'bg-slate-800 text-slate-400'}`}
            title="Foto Çeviri"
          >
            <Camera size={16} />
            <span className="text-[10px] font-bold uppercase tracking-wider">Foto</span>
          </button>
          <button 
            onClick={() => {
              setIsKeyboardVisible(!isKeyboardVisible);
              triggerHaptic();
            }} 
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full transition-all ${isKeyboardVisible ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-400'}`}
            title="Klavye Girişi"
          >
            <Keyboard size={16} />
            <span className="text-[10px] font-bold uppercase tracking-wider">Yaz</span>
          </button>
          {!isListenModeActive && viewMode !== 'archive' && (
            <button onClick={() => setViewMode(viewMode === 'chat' ? 'split' : 'chat')} className={`p-2 rounded-full transition-all ${viewMode === 'split' ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-400'}`}>
              {viewMode === 'chat' ? <SplitSquareVertical size={18} /> : <MessageSquare size={18} />}
            </button>
          )}
          <button onClick={() => setShowSettings(true)} className="p-2 rounded-full bg-slate-800 text-slate-400 hover:text-white">
            <Settings size={20} />
          </button>
        </div>
      </header>

      {/* ERROR BANNER */}
      {error && (
        <div className="z-50 bg-red-600 text-white px-4 py-2 flex items-center justify-between animate-in slide-in-from-top duration-300">
          <div className="flex items-center gap-2 text-sm font-medium">
            <Zap size={16} fill="currentColor" />
            <span>{error}</span>
          </div>
          <button onClick={() => setError(null)} className="p-1 hover:bg-white/20 rounded-full transition-colors">
            <X size={16} />
          </button>
        </div>
      )}

      {/* MAIN CONTENT */}
      <main className="flex-1 overflow-hidden relative flex flex-col z-10">
        {viewMode === 'archive' ? (
          <div className="flex-1 overflow-y-auto p-4 space-y-4 no-scrollbar">
            {openedArchive ? (
              <div className="space-y-4 animate-fade-in">
                <button onClick={() => setOpenedArchive(null)} className="text-blue-400 text-sm flex items-center gap-1 mb-2"><ChevronRight className="rotate-180" size={16}/> Arşive Dön</button>
                <div className="bg-slate-900/50 p-5 rounded-2xl border border-slate-800 flex justify-between items-center">
                   <div>
                     <h3 className="font-bold text-lg">{new Date(openedArchive.date).toLocaleDateString()}</h3>
                     <p className="text-xs text-slate-500">{getLangDetails(openedArchive.targetLang).name} Tercümesi</p>
                   </div>
                   <button onClick={() => deleteArchivedSession(openedArchive.id)} className="p-3 bg-red-500/10 text-red-400 rounded-xl"><Trash2 size={20}/></button>
                </div>
                {openedArchive.messages.map((m, i) => (
                  <div key={i} className={`flex flex-col ${m.role === 'user' ? 'items-start' : 'items-end'}`}>
                    <div className={`max-w-[85%] p-4 rounded-2xl ${m.role === 'user' ? 'bg-slate-800/80 text-slate-300' : 'bg-blue-600 text-white'}`}>
                      <p className="text-[15px] leading-relaxed">{m.text}</p>
                      {m.role === 'model' && (
                        <button 
                          onClick={() => handlePlaySpeech(m.text, `archive-${i}`)}
                          className={`mt-2 p-1.5 rounded-lg transition-all ${isSpeechPlaying === `archive-${i}` ? 'bg-white/20 text-white animate-pulse' : 'hover:bg-white/10 text-white/70'}`}
                        >
                          <Volume2 size={16} />
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="space-y-3">
                <h2 className="text-xl font-bold px-1 mb-4 flex items-center gap-2"><FolderOpen size={20} className="text-blue-400"/> Kayıtlı Oturumlar</h2>
                {savedSessions.length === 0 ? (
                  <div className="py-20 text-center opacity-30 flex flex-col items-center"><ScrollText size={48} /><p className="mt-4">Henüz kayıt yok</p></div>
                ) : (
                  savedSessions.map(s => (
                    <div key={s.id} onClick={() => setOpenedArchive(s)} className="bg-slate-900/50 p-5 rounded-2xl border border-slate-800 cursor-pointer hover:border-slate-600 transition-all group">
                      <div className="flex justify-between items-start mb-2">
                        <h4 className="font-bold text-slate-200">{new Date(s.date).toLocaleDateString()}</h4>
                        <span className="text-[10px] bg-slate-800 px-2 py-0.5 rounded uppercase tracking-widest text-slate-500">{getLangDetails(s.targetLang).short}</span>
                      </div>
                      <p className="text-xs text-slate-500 line-clamp-2 leading-relaxed">{s.preview}</p>
                    </div>
                  ))
                )}
              </div>
            )}
          </div>
        ) : viewMode === 'photo' ? (
          <div className="flex-1 relative bg-black flex flex-col">
            {!capturedImage ? (
              <div className="flex-1 relative overflow-hidden">
                <video 
                  ref={videoRef} 
                  autoPlay 
                  playsInline 
                  muted
                  className="absolute inset-0 w-full h-full object-cover"
                />
                <div className="absolute inset-0 border-[2px] border-white/20 m-10 rounded-3xl pointer-events-none flex items-center justify-center">
                   <div className="w-10 h-10 border-t-2 border-l-2 border-white absolute top-0 left-0 rounded-tl-lg"></div>
                   <div className="w-10 h-10 border-t-2 border-r-2 border-white absolute top-0 right-0 rounded-tr-lg"></div>
                   <div className="w-10 h-10 border-b-2 border-l-2 border-white absolute bottom-0 left-0 rounded-bl-lg"></div>
                   <div className="w-10 h-10 border-b-2 border-r-2 border-white absolute bottom-0 right-0 rounded-br-lg"></div>
                </div>
                <div className="absolute bottom-10 left-0 right-0 flex justify-center px-6">
                  <button 
                    onClick={handleCapture}
                    className="w-20 h-20 rounded-full border-4 border-white flex items-center justify-center bg-white/10 backdrop-blur-md active:scale-90 transition-transform"
                  >
                    <div className="w-14 h-14 rounded-full bg-white"></div>
                  </button>
                </div>
              </div>
            ) : (
              <div className="flex-1 relative flex flex-col">
                <img src={capturedImage} className="flex-1 object-cover" alt="Captured" />
                <div className="absolute inset-0 bg-black/40 backdrop-blur-[2px] flex flex-col items-center justify-center p-8 text-center">
                  {isAnalyzing ? (
                    <div className="space-y-4 flex flex-col items-center">
                      <div className="w-16 h-16 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin"></div>
                      <p className="text-emerald-400 font-bold animate-pulse tracking-widest uppercase text-xs">Ürün Analiz Ediliyor...</p>
                    </div>
                  ) : analysisResult ? (
                    <div className="bg-slate-900/90 border border-white/10 p-8 rounded-[2.5rem] shadow-2xl max-w-sm w-full animate-in zoom-in-95 duration-300">
                      <div className="w-12 h-12 bg-emerald-500/20 text-emerald-400 rounded-2xl flex items-center justify-center mb-6 mx-auto">
                        <Sparkles size={24} />
                      </div>
                      <h3 className="text-2xl font-bold text-white mb-2">{analysisResult.translation}</h3>
                      <p className="text-slate-400 text-sm leading-relaxed mb-8">{analysisResult.info}</p>
                      <div className="flex gap-3">
                        <button 
                          onClick={() => setCapturedImage(null)}
                          className="flex-1 py-4 bg-slate-800 rounded-2xl font-bold text-sm flex items-center justify-center gap-2"
                        >
                          <RefreshCw size={16} /> Tekrar
                        </button>
                        <button 
                          onClick={() => handlePlaySpeech(analysisResult.translation + ". " + analysisResult.info, 'photo-result')}
                          className={`flex-1 py-4 bg-emerald-600 rounded-2xl font-bold text-sm flex items-center justify-center gap-2 ${isSpeechPlaying === 'photo-result' ? 'animate-pulse' : ''}`}
                        >
                          <Volume2 size={16} /> Dinle
                        </button>
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>
            )}
            <canvas ref={canvasRef} className="hidden" />
          </div>
        ) : viewMode === 'listen' ? (
          <div className="flex-1 overflow-y-auto px-6 py-10 space-y-10 no-scrollbar">
             {realtimeOutput && (
               <div className="animate-fade-in">
                 <span className="text-[10px] uppercase font-bold tracking-widest text-orange-500 block mb-2">Çevriliyor...</span>
                 <p className="text-4xl font-bold text-orange-200 leading-tight">{realtimeOutput}</p>
               </div>
             )}
             {[...messages].filter(m => m.role === 'model').reverse().map(m => (
               <div key={m.id} className="animate-fade-in-up">
                 <span className="text-[10px] uppercase font-bold tracking-widest text-orange-500 block mb-2">Çeviri ({sourceDetails.short})</span>
                 <p className="text-4xl font-bold leading-tight drop-shadow-sm">{m.text}</p>
               </div>
             ))}
             {realtimeInput && (
               <div className="opacity-40 animate-pulse">
                 <span className="text-[10px] uppercase font-bold tracking-widest text-orange-400 block mb-2">Dinleniyor ({targetDetails.short})</span>
                 <p className="text-2xl italic leading-relaxed">{realtimeInput}...</p>
               </div>
             )}
 
          </div>
        ) : viewMode === 'split' ? (
          <div className="flex-1 flex flex-col">
            <div className="flex-1 bg-slate-900/50 flex items-center justify-center p-6 rotate-180 border-b border-white/5 overflow-y-auto">
               <div className="text-center space-y-2 w-full max-w-2xl">
                 <span className="text-[10px] font-bold text-emerald-500/50 uppercase tracking-[0.3em]">{targetDetails.name}</span>
                 <p className="text-2xl sm:text-4xl font-bold text-emerald-400 leading-tight break-words">{realtimeOutput || messages.filter(m => m.role === 'model').pop()?.text || '...'}</p>
                 {(realtimeOutput || messages.filter(m => m.role === 'model').pop()?.text) && (
                    <button 
                      onClick={() => handlePlaySpeech(realtimeOutput || messages.filter(m => m.role === 'model').pop()?.text || '', 'split-target')}
                      className={`mt-4 p-3 rounded-full bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20 transition-all ${isSpeechPlaying === 'split-target' ? 'animate-pulse' : ''}`}
                    >
                      <Volume2 size={24} />
                    </button>
                  )}
               </div>
            </div>
            <div className="flex-1 bg-slate-950/50 flex items-center justify-center p-6 overflow-y-auto">
               <div className="text-center space-y-2 w-full max-w-2xl">
                 <span className="text-[10px] font-bold text-blue-500/50 uppercase tracking-[0.3em]">{sourceDetails.name}</span>
                 <p className="text-2xl sm:text-4xl font-bold text-white leading-tight break-words">{realtimeInput || messages.filter(m => m.role === 'user').pop()?.text || '...'}</p>
               </div>
            </div>
          </div>
        ) : (
          <div className="flex-1 overflow-y-auto px-5 py-8 space-y-8 no-scrollbar">
            {messages.length === 0 && !realtimeInput && (
              <div className="h-full flex flex-col items-center justify-center opacity-10 space-y-4 py-20">
                <Globe size={80} />
                <p className="text-center font-bold">KONUŞMAYA BAŞLA</p>
              </div>
            )}
            {messages.map(m => (
              <div key={m.id} className={`flex flex-col ${m.role === 'user' ? 'items-start' : 'items-end'}`}>
                <span className="text-[10px] text-slate-500 mb-1 px-1">{m.role === 'user' ? 'Siz' : 'Tercüman'}</span>
                <div className={`max-w-[85%] p-4 rounded-2xl shadow-sm ${m.role === 'user' ? 'bg-slate-800/80 text-slate-200 rounded-tl-sm' : 'bg-emerald-600 text-white rounded-tr-sm'}`}>
                  <p className="text-[15px] leading-relaxed">{m.text}</p>
                  {m.role === 'model' && (
                    <button 
                      onClick={() => handlePlaySpeech(m.text, m.id)}
                      className={`mt-2 p-1.5 rounded-lg transition-all ${isSpeechPlaying === m.id ? 'bg-white/20 text-white animate-pulse' : 'hover:bg-white/10 text-white/70'}`}
                    >
                      <Volume2 size={16} />
                    </button>
                  )}
                </div>
              </div>
            ))}
            {realtimeInput && (
              <div className="flex flex-col items-start opacity-60">
                <span className="text-[10px] text-blue-400 mb-1 px-1 animate-pulse">ALGILANIYOR...</span>
                <div className="max-w-[85%] p-4 bg-slate-800/30 rounded-2xl rounded-tl-sm border border-slate-700 border-dashed text-slate-300">
                  {realtimeInput}
                </div>
              </div>
            )}
            {realtimeOutput && (
              <div className="flex flex-col items-end opacity-60">
                <span className="text-[10px] text-emerald-400 mb-1 px-1 animate-pulse">ÇEVRİLİYOR...</span>
                <div className="max-w-[85%] p-4 bg-emerald-900/20 rounded-2xl rounded-tr-sm border border-emerald-500/30 border-dashed text-emerald-200">
                  {realtimeOutput}
                </div>
              </div>
            )}
          </div>
        )}

        {/* CONTROLS */}
        <div className={`relative w-full p-6 backdrop-blur-2xl border-t border-white/5 pb-[calc(1.5rem+env(safe-area-inset-bottom,20px))] transition-colors duration-1000 ${isListenModeActive ? 'bg-orange-950/90' : 'bg-slate-950/90'}`}>
          <div className="h-6 w-full flex items-center justify-center mb-6">
            {isConnected && <AudioVisualizer analyser={inputAnalyserRef.current} isActive={true} color={isListenModeActive ? '#f97316' : '#10b981'} />}
          </div>
          
          {/* TEXT INPUT TRANSLATION */}
          {isKeyboardVisible && !isListenModeActive && (
            <div className="max-w-md mx-auto mb-6 px-4 animate-in slide-in-from-bottom-4 duration-300">
              <div className="relative flex items-center group">
                <input 
                  type="text"
                  value={textInput}
                  onChange={e => setTextInput(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleTextSubmit()}
                  placeholder={`${sourceDetails.name} dilinde yazın...`}
                  className="w-full bg-white/5 border border-white/10 rounded-2xl py-4 pl-5 pr-14 text-white placeholder:text-slate-500 focus:ring-2 focus:ring-blue-500/50 focus:bg-white/10 outline-none transition-all shadow-xl"
                />
                <button 
                  onClick={handleTextSubmit}
                  disabled={!textInput.trim() || isTextTranslating}
                  className={`absolute right-2 w-10 h-10 rounded-xl flex items-center justify-center transition-all ${textInput.trim() && !isTextTranslating ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20' : 'text-slate-400/30'}`}
                >
                  {isTextTranslating ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
                </button>
              </div>
            </div>
          )}

          <div className="max-w-md mx-auto flex items-center justify-between gap-4">
            <button 
              onClick={() => !isConnected && setShowLangSelector('source')}
              className={`w-14 h-14 bg-slate-800/50 border border-slate-700 rounded-2xl flex items-center justify-center text-2xl shadow-lg active:scale-95 transition-transform ${isConnected ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              {sourceDetails.flag}
            </button>
            
            <div className="flex gap-4 items-end -mt-6">
              <div className="relative group">
                {isListenModeActive && <div className="absolute inset-0 rounded-full animate-ping bg-orange-500/40"></div>}
                <button 
                  onMouseDown={isNoiseMode ? () => setIsHoldingMic(true) : undefined}
                  onMouseUp={isNoiseMode ? () => setIsHoldingMic(false) : undefined}
                  onTouchStart={isNoiseMode ? () => setIsHoldingMic(true) : undefined}
                  onTouchEnd={isNoiseMode ? () => setIsHoldingMic(false) : undefined}
                  onClick={() => !isNoiseMode && startLiveSession('listen')} 
                  className={`relative w-12 h-12 rounded-full flex items-center justify-center shadow-xl transition-all duration-300 transform active:scale-95 ${isListenModeActive ? 'bg-orange-500 text-white scale-110' : 'bg-slate-800 text-slate-500 hover:bg-slate-700'}`}
                >
                  {isConnecting && isListenModeActive ? <Loader2 size={20} className="animate-spin" /> : <Ear size={20} />}
                </button>
              </div>

              <div className="relative">
                {isConnected && !isListenModeActive && <div className="absolute inset-0 rounded-full animate-ping bg-emerald-500/40"></div>}
                <button 
                  onMouseDown={isNoiseMode ? () => setIsHoldingMic(true) : undefined}
                  onMouseUp={isNoiseMode ? () => setIsHoldingMic(false) : undefined}
                  onTouchStart={isNoiseMode ? () => setIsHoldingMic(true) : undefined}
                  onTouchEnd={isNoiseMode ? () => setIsHoldingMic(false) : undefined}
                  onClick={() => !isNoiseMode && startLiveSession('bidirectional')} 
                  className={`relative w-16 h-16 rounded-full flex items-center justify-center shadow-2xl transition-all duration-300 transform active:scale-95 z-10 ${isConnected && !isListenModeActive ? 'bg-red-600 scale-105' : 'bg-white text-slate-950'}`}
                >
                  {isConnecting && !isListenModeActive ? <Loader2 size={24} className="animate-spin text-slate-400" /> : isConnected && !isListenModeActive ? <Square size={22} fill="currentColor" /> : <Mic size={28} />}
                </button>
              </div>
            </div>

            <button 
              onClick={() => !isConnected && setShowLangSelector('target')}
              className={`w-14 h-14 bg-slate-800/50 border border-slate-700 rounded-2xl flex items-center justify-center text-2xl shadow-lg active:scale-95 transition-transform ${isConnected ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              {targetDetails.flag}
            </button>
          </div>
        </div>
      </main>

      {/* LANGUAGE SELECTOR MODAL */}
      {showLangSelector && (
        <div className="fixed inset-0 z-[110] bg-black/80 backdrop-blur-md flex items-center justify-center p-6 animate-fade-in">
          <div className="bg-slate-900 border border-slate-800 w-full max-w-sm rounded-3xl shadow-2xl p-6">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-xl font-bold">{showLangSelector === 'source' ? 'Kaynak Dil' : 'Hedef Dil'}</h3>
              <button onClick={() => setShowLangSelector(null)} className="p-2 bg-slate-800 rounded-full"><X size={18} /></button>
            </div>
            <div className="grid grid-cols-2 gap-3 max-h-[60vh] overflow-y-auto no-scrollbar pr-1">
              {LANGUAGE_META.map((lang) => (
                <button
                  key={lang.code}
                  onClick={() => {
                    if (showLangSelector === 'source') setSourceLang(lang.code);
                    else setTargetLang(lang.code);
                    setShowLangSelector(null);
                    triggerHaptic();
                  }}
                  className={`flex items-center gap-3 p-4 rounded-2xl border transition-all ${
                    (showLangSelector === 'source' ? sourceLang : targetLang) === lang.code
                      ? 'bg-blue-600 border-blue-500 text-white'
                      : 'bg-slate-800 border-slate-700 text-slate-300 hover:border-slate-500'
                  }`}
                >
                  <span className="text-2xl">{lang.flag}</span>
                  <span className="font-medium text-sm">{lang.name}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* SAVE MODAL */}
      {showSaveModal && (
          <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-6 animate-fade-in">
              <div className="bg-slate-900 border border-slate-800 w-full max-w-sm rounded-3xl shadow-2xl p-8 text-center">
                  <div className="w-16 h-16 bg-blue-600/20 text-blue-400 rounded-full flex items-center justify-center mb-6 mx-auto"><Save size={32} /></div>
                  <h3 className="text-2xl font-bold mb-2">Oturumu Kaydet?</h3>
                  <p className="text-slate-500 text-sm mb-8 leading-relaxed">Bu görüşmedeki çevirileri daha sonra incelemek için arşive eklemek ister misiniz?</p>
                  <div className="grid grid-cols-2 gap-3">
                      <button onClick={() => { setMessages([]); setShowSaveModal(false); stopConnection(); }} className="py-4 rounded-2xl bg-slate-800 font-bold">Sil</button>
                      <button onClick={saveSession} className="py-4 rounded-2xl bg-blue-600 font-bold">Kaydet</button>
                  </div>
              </div>
          </div>
      )}

      {/* API KEY MODAL */}
      {!Object.values(apiKeys).some(k => !!k) && (
        <div className="fixed inset-0 z-[100] bg-slate-950 flex items-center justify-center p-8 animate-fade-in">
           <div className="w-full max-w-sm text-center">
              <div className="w-20 h-20 bg-blue-600/10 rounded-3xl flex items-center justify-center mx-auto mb-8 shadow-2xl"><Key size={40} className="text-blue-500" /></div>
              <h1 className="text-3xl font-bold mb-2">Hoş Geldiniz</h1>
              <p className="text-slate-500 mb-8 text-sm">Devam etmek için bir yapay zeka sağlayıcısı seçin ve API anahtarınızı girin.</p>
              
              <div className="flex gap-2 mb-6">
                {Object.values(AIProvider).map(p => (
                  <button
                    key={p}
                    onClick={() => setTempProviderInput(p)}
                    className={`flex-1 py-3 rounded-xl text-xs font-bold transition-all border ${tempProviderInput === p ? 'bg-blue-600 border-blue-500 text-white shadow-lg shadow-blue-900/20' : 'bg-slate-900 border-slate-800 text-slate-400'}`}
                  >
                    {p}
                  </button>
                ))}
              </div>

              <div className="relative mb-6">
                <input 
                  type="password" 
                  value={tempApiKeyInput} 
                  onChange={e => setTempApiKeyInput(e.target.value)} 
                  placeholder={`${tempProviderInput} API Anahtarı`} 
                  className="w-full bg-slate-900 border border-slate-800 rounded-2xl p-5 text-white focus:ring-2 focus:ring-blue-600 outline-none transition-all" 
                />
              </div>

              <button 
                onClick={() => { 
                  if(tempApiKeyInput.trim()) { 
                    const newKeys = { ...apiKeys, [tempProviderInput.toLowerCase()]: tempApiKeyInput.trim() };
                    localStorage.setItem('ai_api_keys', JSON.stringify(newKeys));
                    localStorage.setItem('selected_provider', tempProviderInput);
                    window.location.reload(); 
                  } 
                }} 
                className="w-full bg-blue-600 hover:bg-blue-500 py-5 rounded-2xl font-bold text-lg shadow-xl shadow-blue-900/20 active:scale-[0.98] transition-all"
              >
                Uygulamayı Başlat
              </button>
              
              <div className="mt-8 space-y-2">
                {tempProviderInput === AIProvider.GEMINI && <a href="https://aistudio.google.com/app/apikey" target="_blank" className="block text-slate-500 text-xs hover:text-white transition-colors">Gemini anahtarı oluştur <ExternalLink size={10} className="inline ml-1" /></a>}
                {tempProviderInput === AIProvider.OPENAI && <a href="https://platform.openai.com/api-keys" target="_blank" className="block text-slate-500 text-xs hover:text-white transition-colors">OpenAI anahtarı oluştur <ExternalLink size={10} className="inline ml-1" /></a>}
                {tempProviderInput === AIProvider.ANTHROPIC && <a href="https://console.anthropic.com/settings/keys" target="_blank" className="block text-slate-500 text-xs hover:text-white transition-colors">Anthropic anahtarı oluştur <ExternalLink size={10} className="inline ml-1" /></a>}
              </div>
           </div>
        </div>
      )}

      {/* SETTINGS SHEET */}
      {showSettings && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-end justify-center animate-fade-in" onClick={() => setShowSettings(false)}>
           <div className="bg-slate-900 w-full max-w-lg rounded-t-[2.5rem] p-8 space-y-8 shadow-2xl overflow-y-auto max-h-[90vh] pb-20 no-scrollbar" onClick={e => e.stopPropagation()}>
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold flex items-center gap-2"><Settings className="text-blue-400" /> Ayarlar</h2>
                <button onClick={() => setShowSettings(false)} className="p-2 bg-slate-800 rounded-full"><X /></button>
              </div>
              
              <button onClick={() => { setViewMode('archive'); setShowSettings(false); }} className="w-full p-5 bg-slate-800/50 hover:bg-slate-800 border border-slate-700/50 rounded-2xl flex items-center justify-between transition-all group">
                <div className="flex items-center gap-4">
                  <div className="p-3 bg-blue-600/20 rounded-xl group-hover:bg-blue-600/40 transition-colors"><FolderOpen size={24} className="text-blue-400" /></div>
                  <div className="text-left">
                    <p className="font-bold">Kayıtlı Oturumlar</p>
                    <p className="text-xs text-slate-500">Geçmiş görüşmeleri incele</p>
                  </div>
                </div>
                <ChevronRight size={20} className="text-slate-600" />
              </button>

              <div className="space-y-4">
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest px-1">Yapay Zeka Modeli</h3>
                <div className="bg-slate-800/30 p-5 rounded-3xl border border-slate-800 space-y-6">
                  <div className="flex gap-2">
                    {Object.values(AIProvider).map(p => (
                      <button
                        key={p}
                        onClick={() => {
                          setSelectedProvider(p);
                          localStorage.setItem('selected_provider', p);
                          triggerHaptic();
                        }}
                        className={`flex-1 py-3 rounded-xl text-xs font-bold transition-all border ${selectedProvider === p ? 'bg-blue-600 border-blue-500 text-white shadow-lg' : 'bg-slate-900 border-slate-800 text-slate-400'}`}
                      >
                        {p}
                      </button>
                    ))}
                  </div>
                  
                  <div className="space-y-4">
                    {Object.values(AIProvider).map(p => (
                      <div key={p} className="relative">
                        <label className="text-[10px] text-slate-500 uppercase font-bold mb-1 block px-1">{p} API Anahtarı</label>
                        <input 
                          type="password"
                          value={apiKeys[p.toLowerCase() as keyof APIKeys] || ''}
                          onChange={e => {
                            const newKeys = { ...apiKeys, [p.toLowerCase()]: e.target.value };
                            setApiKeys(newKeys);
                            localStorage.setItem('ai_api_keys', JSON.stringify(newKeys));
                          }}
                          placeholder={`${p} anahtarını girin`}
                          className="w-full bg-slate-900 border border-slate-800 rounded-xl p-3 text-sm text-white focus:ring-1 focus:ring-blue-600 outline-none"
                        />
                      </div>
                    ))}
                  </div>
                  
                  {selectedProvider !== AIProvider.GEMINI && (
                    <div className="p-3 bg-amber-500/10 border border-amber-500/20 rounded-xl flex gap-3 items-start">
                      <Zap size={16} className="text-amber-500 shrink-0 mt-0.5" />
                      <p className="text-[10px] text-amber-200/70 leading-relaxed">
                        <strong>Not:</strong> Sesli canlı çeviri (Canlı Mod) şu an sadece Gemini ile çalışmaktadır. Diğer modeller sadece klavye girişi ile yapılan çevirilerde kullanılabilir.
                      </p>
                    </div>
                  )}
                </div>
              </div>

              <div className="space-y-4">
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest px-1">Özellikler</h3>
                <div className="bg-slate-800/30 p-5 rounded-3xl border border-slate-800 space-y-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-bold text-slate-200">Gürültü Modu (PTT)</p>
                      <p className="text-xs text-slate-500">Sadece basılı tuttuğunuzda dinler</p>
                    </div>
                    <button onClick={() => setIsNoiseMode(!isNoiseMode)} className={`w-14 h-8 rounded-full p-1 transition-colors ${isNoiseMode ? 'bg-emerald-600' : 'bg-slate-700'}`}>
                      <div className={`w-6 h-6 rounded-full bg-white transition-transform ${isNoiseMode ? 'translate-x-6' : ''}`}></div>
                    </button>
                  </div>
                </div>

                {/* Voice Selection */}
                <div className="bg-slate-800/30 p-5 rounded-3xl border border-slate-800 space-y-4">
                   <div className="flex items-center gap-2 px-1">
                       <User size={16} className="text-blue-400" />
                       <span className="font-bold text-slate-200 text-sm uppercase tracking-widest">Ses Tercihi</span>
                   </div>
                   <div className="grid grid-cols-2 gap-3">
                       <button 
                         onClick={() => handleVoiceChange('female')}
                         className={`py-4 rounded-2xl flex flex-col items-center justify-center gap-2 transition-all border ${voiceType === 'female' ? 'bg-blue-600 border-blue-500 text-white shadow-lg' : 'bg-slate-900 border-slate-800 text-slate-400'}`}
                       >
                           <span className="text-2xl">👩</span>
                           <span className="font-bold text-xs">Kadın</span>
                       </button>
                       <button 
                         onClick={() => handleVoiceChange('male')}
                         className={`py-4 rounded-2xl flex flex-col items-center justify-center gap-2 transition-all border ${voiceType === 'male' ? 'bg-blue-600 border-blue-500 text-white shadow-lg' : 'bg-slate-900 border-slate-800 text-slate-400'}`}
                       >
                           <span className="text-2xl">👨</span>
                           <span className="font-bold text-xs">Erkek</span>
                       </button>
                   </div>
                   <p className="text-[10px] text-slate-500 px-1 italic">Çevirmen sesi (Sadece Gemini Live modunda geçerlidir).</p>
                </div>
              </div>

              <div className="space-y-4">
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest px-1">Bilgi</h3>
                <div className="grid grid-cols-2 gap-3">
                  <button onClick={() => setShowUpdates(true)} className="p-5 bg-slate-800/50 hover:bg-slate-800 border border-slate-700/50 rounded-2xl flex flex-col items-center gap-2 transition-all group">
                    <div className="p-3 bg-blue-600/20 rounded-xl group-hover:bg-blue-600/40 transition-colors"><History size={20} className="text-blue-400" /></div>
                    <span className="text-xs font-bold">Güncellemeler</span>
                  </button>
                  <button onClick={() => setShowGuide(true)} className="p-5 bg-slate-800/50 hover:bg-slate-800 border border-slate-700/50 rounded-2xl flex flex-col items-center gap-2 transition-all group">
                    <div className="p-3 bg-emerald-600/20 rounded-xl group-hover:bg-emerald-600/40 transition-colors"><BookOpen size={20} className="text-emerald-400" /></div>
                    <span className="text-xs font-bold">Kullanım Kılavuzu</span>
                  </button>
                </div>
              </div>

              <div className="space-y-4">
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest px-1">Güvenlik</h3>
                <button 
                  onClick={() => { localStorage.removeItem('ai_api_keys'); localStorage.removeItem('gemini_api_key'); window.location.reload(); }} 
                  className="w-full p-5 bg-red-500/10 hover:bg-red-500/20 text-red-400 rounded-2xl font-bold flex items-center justify-center gap-2 transition-colors border border-red-500/10"
                >
                  <LogOut size={20} /> Tüm Verileri Temizle ve Çıkış Yap
                </button>
              </div>
              
              <div className="text-center text-[10px] text-slate-600 pt-4 uppercase tracking-[0.2em]">Ai Live Translate v1.0 • Stable Build</div>
           </div>
        </div>
      )}

      {/* UPDATES MODAL */}
      {showUpdates && (
        <div className="fixed inset-0 z-[110] bg-black/80 backdrop-blur-md flex items-center justify-center p-6 animate-fade-in">
          <div className="bg-slate-900 border border-slate-800 w-full max-w-lg rounded-[2.5rem] shadow-2xl flex flex-col max-h-[80vh]">
            <div className="p-8 border-b border-slate-800 flex justify-between items-center">
              <h2 className="text-2xl font-bold flex items-center gap-3"><History className="text-blue-400" /> Güncellemeler</h2>
              <button onClick={() => setShowUpdates(false)} className="p-2 bg-slate-800 rounded-full hover:bg-slate-700 transition-colors"><X /></button>
            </div>
            <div className="p-8 overflow-y-auto space-y-8 no-scrollbar">
              <div className="relative pl-8 border-l-2 border-blue-600/30 space-y-2">
                <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-blue-600 border-4 border-slate-900"></div>
                <div className="flex items-center gap-2">
                  <span className="text-blue-400 font-bold">v1.4</span>
                  <span className="text-[10px] text-slate-500 uppercase tracking-widest">12 Mart 2026</span>
                </div>
                <h4 className="font-bold text-lg">Foto Çeviri Özelliği</h4>
                <ul className="text-sm text-slate-400 space-y-2 list-disc pl-4">
                  <li>Kamera ile ürün/metin fotoğrafı çekip anlık çeviri yapma özelliği eklendi.</li>
                  <li>Görsel analizi ile ürün hakkında kısa bilgi alma özelliği getirildi.</li>
                </ul>
              </div>

              <div className="relative pl-8 border-l-2 border-blue-600/30 space-y-2">
                <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-blue-600 border-4 border-slate-900"></div>
                <div className="flex items-center gap-2">
                  <span className="text-blue-400 font-bold">v1.3</span>
                  <span className="text-[10px] text-slate-500 uppercase tracking-widest">11 Mart 2026</span>
                </div>
                <h4 className="font-bold text-lg">Bilgi ve Rehber</h4>
                <ul className="text-sm text-slate-400 space-y-2 list-disc pl-4">
                  <li>Kullanım Kılavuzu ve Güncellemeler bölümleri eklendi.</li>
                  <li>Ses Tercihi (Kadın/Erkek) özelliği getirildi.</li>
                  <li>Haptik geri bildirim desteği eklendi.</li>
                  <li>Yazım hataları düzeltildi ("Developer" imzası).</li>
                </ul>
              </div>

              <div className="relative pl-8 border-l-2 border-slate-800 space-y-2">
                <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-slate-700 border-4 border-slate-900"></div>
                <div className="flex items-center gap-2">
                  <span className="text-slate-400 font-bold">v1.2</span>
                  <span className="text-[10px] text-slate-500 uppercase tracking-widest">11 Mart 2026</span>
                </div>
                <h4 className="font-bold text-lg">Çoklu Model Desteği</h4>
                <ul className="text-sm text-slate-400 space-y-2 list-disc pl-4">
                  <li>OpenAI (GPT-4o) ve Anthropic (Claude 3.5 Sonnet) desteği eklendi.</li>
                  <li>Çoklu API anahtarı yönetimi ve sağlayıcı seçimi getirildi.</li>
                </ul>
              </div>
              
              <div className="relative pl-8 border-l-2 border-slate-800 space-y-2">
                <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-slate-700 border-4 border-slate-900"></div>
                <div className="flex items-center gap-2">
                  <span className="text-slate-400 font-bold">v1.1</span>
                  <span className="text-[10px] text-slate-500 uppercase tracking-widest">11 Mart 2026</span>
                </div>
                <h4 className="font-bold text-lg">Yenilenen Kimlik</h4>
                <ul className="text-sm text-slate-400 space-y-2 list-disc pl-4">
                  <li>Uygulama ismi "Ai Live Translate" olarak güncellendi.</li>
                  <li>"Developer by Ali TELLIOGLU" imzası eklendi.</li>
                  <li>Yeni logo ve görsel düzenlemeler yapıldı.</li>
                </ul>
              </div>

              <div className="relative pl-8 border-l-2 border-slate-800 space-y-2">
                <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-slate-700 border-4 border-slate-900"></div>
                <div className="flex items-center gap-2">
                  <span className="text-slate-400 font-bold">v1.0</span>
                  <span className="text-[10px] text-slate-500 uppercase tracking-widest">Mart 2026</span>
                </div>
                <h4 className="font-bold text-lg">Lansman</h4>
                <ul className="text-sm text-slate-400 space-y-2 list-disc pl-4">
                  <li>Gemini Live API ile gerçek zamanlı simultane çeviri.</li>
                  <li>Çevrimdışı mod ve dil paketleri desteği.</li>
                  <li>Sesli ve yazılı çeviri özellikleri.</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* GUIDE MODAL */}
      {showGuide && (
        <div className="fixed inset-0 z-[110] bg-black/80 backdrop-blur-md flex items-center justify-center p-6 animate-fade-in">
          <div className="bg-slate-900 border border-slate-800 w-full max-w-lg rounded-[2.5rem] shadow-2xl flex flex-col max-h-[80vh]">
            <div className="p-8 border-b border-slate-800 flex justify-between items-center">
              <h2 className="text-2xl font-bold flex items-center gap-3"><BookOpen className="text-emerald-400" /> Kullanım Kılavuzu</h2>
              <button onClick={() => setShowGuide(false)} className="p-2 bg-slate-800 rounded-full hover:bg-slate-700 transition-colors"><X /></button>
            </div>
            <div className="p-8 overflow-y-auto space-y-8 no-scrollbar">
              <section className="space-y-3">
                <h4 className="text-emerald-400 font-bold uppercase text-xs tracking-widest">Foto Çeviri</h4>
                <p className="text-sm text-slate-300 leading-relaxed">
                  Üst menüdeki "Foto" butonuna basarak kamerayı açın. Ürünün veya metnin fotoğrafını çekin. Yapay zeka görseli analiz ederek metni çevirecek ve ürün hakkında kısa bilgi verecektir.
                </p>
              </section>

              <section className="space-y-3">
                <h4 className="text-emerald-400 font-bold uppercase text-xs tracking-widest">Sesli Çeviri (Canlı Mod)</h4>
                <p className="text-sm text-slate-300 leading-relaxed">
                  Ana ekrandaki büyük mikrofon butonuna basarak canlı çeviriyi başlatın. Konuştuğunuzda sistem sesinizi otomatik olarak algılar ve saniyeler içinde hedef dile çevirerek sesli olarak seslendirir.
                </p>
                <div className="p-3 bg-slate-800/50 rounded-xl text-xs text-slate-400 italic border-l-2 border-emerald-500">
                  İpucu: Gürültülü ortamlarda Ayarlar'dan "Gürültü Modu"nu açarak sadece butona basılı tuttuğunuzda dinlemesini sağlayabilirsiniz.
                </div>
              </section>

              <section className="space-y-3">
                <h4 className="text-emerald-400 font-bold uppercase text-xs tracking-widest">Yazılı Çeviri</h4>
                <p className="text-sm text-slate-300 leading-relaxed">
                  Üst menüdeki "Yaz" butonuna basarak klavyeyi açabilirsiniz. Metninizi yazıp gönderdiğinizde seçili olan yapay zeka modeli (Gemini, OpenAI veya Anthropic) tarafından çeviri yapılır.
                </p>
              </section>

              <section className="space-y-3">
                <h4 className="text-emerald-400 font-bold uppercase text-xs tracking-widest">Yüz Yüze (Split) Modu</h4>
                <p className="text-sm text-slate-300 leading-relaxed">
                  Üst menüdeki kare ikonuna basarak ekranı ikiye bölebilirsiniz. Bu mod, masada karşılıklı oturan kişiler için tasarlanmıştır. Üst kısım karşıdaki kişiye göre 180 derece ters döner, böylece her iki taraf da çeviriyi kendi yönünden okuyabilir.
                </p>
              </section>

              <section className="space-y-3">
                <h4 className="text-emerald-400 font-bold uppercase text-xs tracking-widest">Yapay Zeka Modelleri</h4>
                <p className="text-sm text-slate-300 leading-relaxed">
                  Ayarlar menüsünden çeviri yapacak beyni seçebilirsiniz. Gemini Live API en hızlı sesli deneyimi sunarken, OpenAI ve Anthropic modelleri yazılı çevirilerde alternatif zeka seviyeleri sunar.
                </p>
              </section>

              <section className="space-y-3">
                <h4 className="text-emerald-400 font-bold uppercase text-xs tracking-widest">Ses Tercihi</h4>
                <p className="text-sm text-slate-300 leading-relaxed">
                  Ayarlar menüsünden çevirmenin sesini "Kadın" veya "Erkek" olarak değiştirebilirsiniz. Bu ayar Gemini Live modu aktifken geçerlidir ve çevirilerin seslendirilme tonunu belirler.
                </p>
              </section>

              <section className="space-y-3">
                <h4 className="text-emerald-400 font-bold uppercase text-xs tracking-widest">Çevrimdışı Mod</h4>
                <p className="text-sm text-slate-300 leading-relaxed">
                  İnternetiniz olmadığında Ayarlar'dan "Çevrimdışı Mod"u aktif edebilirsiniz. Bunun için önceden ilgili dil paketlerini indirmiş olmanız gerekir.
                </p>
              </section>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;