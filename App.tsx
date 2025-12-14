import React, { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality } from '@google/genai';
import { TargetLanguage, ChatMessage, OfflinePack } from './types';
import { float32To16BitPCM, arrayBufferToBase64, base64ToArrayBuffer, pcm16ToFloat32 } from './utils/audioUtils';
import AudioVisualizer from './components/AudioVisualizer';
import { Mic, Globe, Settings, RotateCcw, Wifi, WifiOff, Download, Check, Trash2, X, Zap, Square, ChevronDown, Sparkles, Loader2, Languages, ArrowRightLeft, ArrowRight, User, SplitSquareVertical, Maximize2, Minimize2, MessageSquare } from 'lucide-react';

// Live API Configuration
const MODEL_NAME = 'gemini-2.5-flash-native-audio-preview-09-2025';
const INPUT_SAMPLE_RATE = 16000;
const OUTPUT_SAMPLE_RATE = 24000;

// Language Metadata with Flags and Locales for Speech Recognition/Synthesis
const LANGUAGE_META = [
  { code: TargetLanguage.TURKISH, name: 'Türkçe', flag: '🇹🇷', short: 'TR', locale: 'tr-TR' },
  { code: TargetLanguage.ENGLISH, name: 'İngilizce', flag: '🇬🇧', short: 'EN', locale: 'en-US' },
  { code: TargetLanguage.DUTCH, name: 'Hollandaca', flag: '🇳🇱', short: 'NL', locale: 'nl-NL' },
  { code: TargetLanguage.ARABIC, name: 'Arapça', flag: '🇸🇦', short: 'AR', locale: 'ar-SA' },
  { code: TargetLanguage.RUSSIAN, name: 'Rusça', flag: '🇷🇺', short: 'RU', locale: 'ru-RU' },
  { code: TargetLanguage.GERMAN, name: 'Almanca', flag: '🇩🇪', short: 'DE', locale: 'de-DE' },
  { code: TargetLanguage.ITALIAN, name: 'İtalyanca', flag: '🇮🇹', short: 'IT', locale: 'it-IT' },
  { code: TargetLanguage.FRENCH, name: 'Fransızca', flag: '🇫🇷', short: 'FR', locale: 'fr-FR' },
];

const getLangDetails = (lang?: TargetLanguage) => {
    if (!lang) return LANGUAGE_META[0];
    return LANGUAGE_META.find(l => l.code === lang) || LANGUAGE_META[0];
};

// Mock Offline Packs
const INITIAL_PACKS: OfflinePack[] = [
  { id: 'tr-en', pair: 'TR ↔ EN', name: 'İngilizce', size: '45 MB', downloaded: false, progress: 0 },
  { id: 'tr-nl', pair: 'TR ↔ NL', name: 'Hollandaca', size: '42 MB', downloaded: false, progress: 0 },
  { id: 'tr-ar', pair: 'TR ↔ AR', name: 'Arapça', size: '50 MB', downloaded: false, progress: 0 },
  { id: 'tr-ru', pair: 'TR ↔ RU', name: 'Rusça', size: '48 MB', downloaded: false, progress: 0 },
  { id: 'tr-de', pair: 'TR ↔ DE', name: 'Almanca', size: '44 MB', downloaded: false, progress: 0 },
  { id: 'tr-it', pair: 'TR ↔ IT', name: 'İtalyanca', size: '43 MB', downloaded: false, progress: 0 },
  { id: 'tr-fr', pair: 'TR ↔ FR', name: 'Fransızca', size: '46 MB', downloaded: false, progress: 0 },
];

const App: React.FC = () => {
  // Independent Source and Target Languages
  const [sourceLang, setSourceLang] = useState<TargetLanguage>(TargetLanguage.TURKISH);
  const [targetLang, setTargetLang] = useState<TargetLanguage>(TargetLanguage.ENGLISH);
  
  // Voice Preference: 'female' | 'male'
  const [voiceType, setVoiceType] = useState<'female' | 'male'>('female');
  
  // UI State for Language Selector
  const [selectorType, setSelectorType] = useState<'source' | 'target' | null>(null);

  // View Modes
  const [viewMode, setViewMode] = useState<'chat' | 'split'>('chat');
  const [focusedMessage, setFocusedMessage] = useState<ChatMessage | null>(null);

  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  
  // Offline Direction State: false = Source->Target, true = Target->Source
  const [isOfflineReverse, setIsOfflineReverse] = useState(false);
  
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [error, setError] = useState<string | null>(null);
  
  // Real-time Display States
  const [realtimeInput, setRealtimeInput] = useState('');
  const [realtimeOutput, setRealtimeOutput] = useState('');
  
  // Offline / Network State
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [forceOfflineMode, setForceOfflineMode] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [offlinePacks, setOfflinePacks] = useState<OfflinePack[]>(INITIAL_PACKS);
  
  // Audio Context Refs
  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const inputAnalyserRef = useRef<AnalyserNode | null>(null);
  const outputAnalyserRef = useRef<AnalyserNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  
  // Logic Refs
  const aiClientRef = useRef<GoogleGenAI | null>(null);
  const sessionPromiseRef = useRef<Promise<any> | null>(null);
  const activeSessionRef = useRef<any>(null);
  const shouldReconnectRef = useRef(false);
  const nextStartTimeRef = useRef<number>(0);
  const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const isTransmittingRef = useRef(false);
  
  // Transcription Buffers
  const currentInputTranscription = useRef('');
  const currentOutputTranscription = useRef('');
  const recognitionRef = useRef<any>(null);

  // 1. Initialize & Network
  useEffect(() => {
    if (process.env.API_KEY) {
      aiClientRef.current = new GoogleGenAI({ apiKey: process.env.API_KEY });
    } else {
      setError("API Anahtarı bulunamadı.");
    }

    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => {
      setIsOnline(false);
      stopConnection();
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    const savedPacks = localStorage.getItem('offlinePacks');
    if (savedPacks) {
      setOfflinePacks(JSON.parse(savedPacks));
    }

    // Load Voice Preference
    const savedVoice = localStorage.getItem('voiceType');
    if (savedVoice === 'male' || savedVoice === 'female') {
        setVoiceType(savedVoice);
    }

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  const isOfflineActive = !isOnline || forceOfflineMode;

  // Haptic Feedback Helper
  const triggerHaptic = () => {
      if (navigator.vibrate) {
          navigator.vibrate(50);
      }
  };

  // 2. Audio Processing
  const processAudioInput = useCallback((inputData: Float32Array) => {
    if (!sessionPromiseRef.current || !isTransmittingRef.current) return;
    
    const pcmData = float32To16BitPCM(inputData);
    const base64Data = arrayBufferToBase64(pcmData);

    sessionPromiseRef.current.then((session) => {
      session.sendRealtimeInput({
        media: {
          mimeType: `audio/pcm;rate=${INPUT_SAMPLE_RATE}`,
          data: base64Data,
        },
      });
    });
  }, []);

  // 3. Stop Connection
  const stopConnection = useCallback(() => {
    setIsConnecting(false);
    setIsConnected(false);
    isTransmittingRef.current = false;
    triggerHaptic();
    
    setRealtimeInput(''); setRealtimeOutput('');
    
    if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(track => track.stop());
        mediaStreamRef.current = null;
    }
    
    if (scriptProcessorRef.current) {
        scriptProcessorRef.current.disconnect();
        scriptProcessorRef.current = null;
    }

    if (inputAudioContextRef.current) {
        inputAudioContextRef.current.close();
        inputAudioContextRef.current = null;
    }
    if (outputAudioContextRef.current) {
        outputAudioContextRef.current.close();
        outputAudioContextRef.current = null;
    }
    
    if (activeSessionRef.current) {
        activeSessionRef.current.close();
        activeSessionRef.current = null;
    }
    sessionPromiseRef.current = null;
    
    if (recognitionRef.current) {
        recognitionRef.current.stop();
    }
  }, []);

  // 4. Toggle Handler (Mic)
  const handleMicToggle = async () => {
    triggerHaptic();
    if (isConnecting) return;

    if (isConnected) {
      stopConnection();
      return;
    }

    setIsConnecting(true);
    setError(null);

    // --- Offline Mode ---
    if (isOfflineActive) {
      setTimeout(() => {
        setIsConnected(true);
        isTransmittingRef.current = true;
        setIsConnecting(false);
        startOfflineRecognition();
      }, 300);
      return;
    }

    // --- Online Mode ---
    if (!aiClientRef.current) {
       setError("API Anahtarı eksik.");
       setIsConnecting(false);
       return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: {
        sampleRate: INPUT_SAMPLE_RATE,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      }});
      mediaStreamRef.current = stream;

      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ 
        sampleRate: INPUT_SAMPLE_RATE 
      });
      inputAudioContextRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      inputAnalyserRef.current = analyser;

      const scriptProcessor = audioCtx.createScriptProcessor(4096, 1, 1);
      scriptProcessorRef.current = scriptProcessor;

      scriptProcessor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        processAudioInput(inputData);
      };

      source.connect(analyser);
      analyser.connect(scriptProcessor);
      scriptProcessor.connect(audioCtx.destination);

      const outCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ 
        sampleRate: OUTPUT_SAMPLE_RATE 
      });
      outputAudioContextRef.current = outCtx;
      const outAnalyser = outCtx.createAnalyser();
      outAnalyser.fftSize = 256;
      outputAnalyserRef.current = outAnalyser;
      outAnalyser.connect(outCtx.destination);

      const sourceDetails = getLangDetails(sourceLang);
      const targetDetails = getLangDetails(targetLang);

      // UPDATED BIDIRECTIONAL SYSTEM PROMPT WITH DYNAMIC PAIRS
      const systemPrompt = `You are an expert bidirectional simultaneous interpreter.
Your task is to facilitate a real-time conversation between a ${sourceDetails.name} speaker and a ${targetDetails.name} speaker.

STRICT INSTRUCTIONS:
1. Bidirectional Translation:
   - Input: ${sourceDetails.name} -> Output: ${targetDetails.name}
   - Input: ${targetDetails.name} -> Output: ${sourceDetails.name}
   
2. Execution:
   - Identify the language spoken automatically (between ${sourceDetails.name} and ${targetDetails.name}).
   - Translate immediately.
   - Do NOT add conversational fillers.
   - Do NOT translate silence.

Output strictly audio of the translation.`;

      // Select Voice
      const voiceName = voiceType === 'female' ? 'Kore' : 'Fenrir';

      const connectPromise = aiClientRef.current.live.connect({
        model: MODEL_NAME,
        config: {
          systemInstruction: systemPrompt,
          responseModalities: [Modality.AUDIO],
          inputAudioTranscription: {},
          outputAudioTranscription: {},
          speechConfig: {
            voiceConfig: {
                prebuiltVoiceConfig: { voiceName: voiceName }
            }
          }
        },
        callbacks: {
          onopen: () => {
            console.log("Connected");
            setIsConnecting(false);
            setIsConnected(true);
            isTransmittingRef.current = true;
            nextStartTimeRef.current = 0;
            setRealtimeInput('');
            setRealtimeOutput('');
          },
          onmessage: (msg: LiveServerMessage) => handleServerMessage(msg),
          onclose: () => {
            console.log("Closed");
            stopConnection();
          },
          onerror: (e) => {
            console.error(e);
            setError("Bağlantı hatası.");
            stopConnection();
          }
        }
      });
      
      sessionPromiseRef.current = connectPromise;
      connectPromise.then(session => {
          activeSessionRef.current = session;
      });

    } catch (e: any) {
      console.error(e);
      setError(`Hata: ${e.message}`);
      stopConnection();
    }
  };

  const handleServerMessage = (msg: LiveServerMessage) => {
    if (msg.serverContent?.inputTranscription) {
       currentInputTranscription.current += msg.serverContent.inputTranscription.text;
       setRealtimeInput(currentInputTranscription.current);
    }
    if (msg.serverContent?.outputTranscription) {
       currentOutputTranscription.current += msg.serverContent.outputTranscription.text;
       setRealtimeOutput(currentOutputTranscription.current);
    }
    if (msg.serverContent?.turnComplete) {
       const inputTx = currentInputTranscription.current.trim();
       const outputTx = currentOutputTranscription.current.trim();
       if (inputTx || outputTx) {
         // Assuming online flow: Input is primarily from Source, Output is primarily Target
         // (In a true bidirectional auto-detect scenario, this is an approximation)
         addMessage('user', inputTx || '...', sourceLang);
         addMessage('model', outputTx || '...', targetLang);
       }
       currentInputTranscription.current = '';
       currentOutputTranscription.current = '';
       setRealtimeInput('');
       setRealtimeOutput('');
    }
    const audioData = msg.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
    if (audioData) {
       playAudioResponse(audioData);
    }
  };

  // 5. Language Change Logic
  const handleLanguageSelect = (newLang: TargetLanguage) => {
      // Prevent selecting same language for source and target
      if (selectorType === 'source') {
          if (newLang === targetLang) {
             // Swap if same
             setTargetLang(sourceLang);
          }
          setSourceLang(newLang);
      } else if (selectorType === 'target') {
          if (newLang === sourceLang) {
             // Swap if same
             setSourceLang(targetLang);
          }
          setTargetLang(newLang);
      }

      setSelectorType(null);

      // Reconnect if online to update system prompt
      if (isConnected && !isOfflineActive) {
          shouldReconnectRef.current = true;
          stopConnection();
      }
  };

  const handleVoiceChange = (type: 'female' | 'male') => {
      setVoiceType(type);
      localStorage.setItem('voiceType', type);
      
      // If connected online, reconnect to apply new voice
      if (isConnected && !isOfflineActive) {
          shouldReconnectRef.current = true;
          stopConnection();
      }
  };

  useEffect(() => {
      if (shouldReconnectRef.current) {
          shouldReconnectRef.current = false;
          handleMicToggle();
      }
  }, [sourceLang, targetLang, voiceType]); // Added voiceType to dependency but handled via shouldReconnect

  // Offline Direction Toggle
  const toggleOfflineDirection = () => {
    if (!isOfflineActive) return; // Only relevant for offline
    if (isConnected) {
        stopConnection();
        // Give a small delay to reset before restarting
        setTimeout(() => {
           setIsOfflineReverse(!isOfflineReverse);
           handleMicToggle(); // Restart immediately
        }, 200);
    } else {
        setIsOfflineReverse(!isOfflineReverse);
    }
  };

  // 6. Offline Logic
  const startOfflineRecognition = () => {
    if (!('webkitSpeechRecognition' in window)) {
        setError("Tarayıcınız ses tanımayı desteklemiyor.");
        return;
    }
    try {
      const Recognition = (window as any).webkitSpeechRecognition;
      const recognition = new Recognition();
      
      const sourceDetails = getLangDetails(sourceLang);
      const targetDetails = getLangDetails(targetLang);
      
      // Determine language based on direction
      // isOfflineReverse FALSE -> Speaking Source
      // isOfflineReverse TRUE -> Speaking Target
      const activeLocale = isOfflineReverse ? targetDetails.locale : sourceDetails.locale;
      
      recognition.lang = activeLocale;
      recognition.continuous = true; 
      recognition.interimResults = true; 

      recognition.onstart = () => {
          setRealtimeInput('');
      };
      
      recognition.onresult = (event: any) => {
        let interim = '';
        let final = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) final += event.results[i][0].transcript;
          else interim += event.results[i][0].transcript;
        }
        setRealtimeInput(final || interim);
      };
      recognition.onerror = (e: any) => console.log(e);
      recognitionRef.current = recognition;
      recognition.start();
    } catch (e) {}
  };

  const stopOfflineRecognition = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      setTimeout(() => {
        const text = realtimeInput.trim();
        if (text) {
           const inputLang = isOfflineReverse ? targetLang : sourceLang;
           addMessage('user', text, inputLang);
           setRealtimeInput('');
           
           // Mock Translation Logic
           setTimeout(() => {
             const sourceDetails = getLangDetails(sourceLang);
             const targetDetails = getLangDetails(targetLang);
             
             let mockTranslation = "";
             let outputLang = targetLang;
             
             if (isOfflineReverse) {
                 // Target -> Source
                 outputLang = sourceLang;
                 mockTranslation = `(${sourceDetails.short} Çevirisi) ${text}`; 
                 speakOffline(text, sourceDetails.locale); 
             } else {
                 // Source -> Target
                 outputLang = targetLang;
                 mockTranslation = `(${targetDetails.short} Çevirisi) ${text}`;
                 speakOffline(text, targetDetails.locale); 
             }
             
             addMessage('model', mockTranslation, outputLang);
           }, 500);
        }
      }, 500);
    }
  };

  const speakOffline = (text: string, locale: string) => {
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text); 
      utterance.lang = locale;
      // Note: Offline mode uses system voice default for locale.
      // Implementing gender specific voice selection for offline is browser dependent and unreliable.
      window.speechSynthesis.speak(utterance);
    }
  };

  // 7. Play Audio
  const playAudioResponse = async (base64Audio: string) => {
    if (!outputAudioContextRef.current || !outputAnalyserRef.current) return;
    const ctx = outputAudioContextRef.current;
    try {
      const arrayBuffer = base64ToArrayBuffer(base64Audio);
      const float32Data = pcm16ToFloat32(arrayBuffer);
      const audioBuffer = ctx.createBuffer(1, float32Data.length, OUTPUT_SAMPLE_RATE);
      audioBuffer.getChannelData(0).set(float32Data);
      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(outputAnalyserRef.current);
      source.start(Math.max(nextStartTimeRef.current, ctx.currentTime));
      nextStartTimeRef.current = Math.max(nextStartTimeRef.current, ctx.currentTime) + audioBuffer.duration;
      source.onended = () => sourcesRef.current.delete(source);
      sourcesRef.current.add(source);
    } catch (err) {}
  };

  const addMessage = (role: 'user' | 'model', text: string, langCode?: TargetLanguage) => {
    setMessages(prev => [...prev, { 
      id: Date.now().toString() + Math.random(), 
      role, 
      text, 
      timestamp: new Date(), 
      isFinal: true,
      langCode: langCode 
    }]);
  };

  const chatContainerRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (chatContainerRef.current) chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
  }, [messages, realtimeInput, realtimeOutput, viewMode]);

  const downloadPack = (id: string) => {
    setOfflinePacks(prev => prev.map(p => p.id === id ? { ...p, progress: 5 } : p));
    let progress = 5;
    const interval = setInterval(() => {
      progress += 10;
      if (progress >= 100) {
        clearInterval(interval);
        setOfflinePacks(prev => {
          const newPacks = prev.map(p => p.id === id ? { ...p, progress: 100, downloaded: true } : p);
          localStorage.setItem('offlinePacks', JSON.stringify(newPacks));
          return newPacks;
        });
      } else {
        setOfflinePacks(prev => prev.map(p => p.id === id ? { ...p, progress } : p));
      }
    }, 200);
  };
  
  const deletePack = (id: string) => {
     setOfflinePacks(prev => {
       const newPacks = prev.map(p => p.id === id ? { ...p, progress: 0, downloaded: false } : p);
       localStorage.setItem('offlinePacks', JSON.stringify(newPacks));
       return newPacks;
     });
  };

  // Get Last Valid Messages for Split View
  const lastUserMsg = messages.slice().reverse().find(m => m.role === 'user');
  const lastModelMsg = messages.slice().reverse().find(m => m.role === 'model');

  const sourceDetails = getLangDetails(sourceLang);
  const targetDetails = getLangDetails(targetLang);

  return (
    <div className="h-[100dvh] w-full bg-slate-950 text-slate-100 flex flex-col font-sans relative overflow-hidden select-none">
      
      {/* Dynamic Background */}
      <div className="absolute inset-0 z-0 pointer-events-none">
        <div className={`absolute top-[-10%] left-[-20%] w-[150vw] h-[60vh] rounded-full blur-[80px] opacity-20 transition-colors duration-1000 ${isOfflineActive ? 'bg-orange-600' : isConnected ? 'bg-emerald-900' : 'bg-blue-900'}`}></div>
        <div className="absolute bottom-[-10%] right-[-20%] w-[150vw] h-[50vh] bg-purple-900/10 rounded-full blur-[100px]"></div>
      </div>

      {/* Mobile Header (Safe Area Top) */}
      <header className="z-20 h-16 flex items-center justify-between px-5 bg-gradient-to-b from-slate-950/80 to-transparent backdrop-blur-sm sticky top-0 pt-[env(safe-area-inset-top)]">
        <div className="flex items-center gap-2.5">
          <div className={`w-8 h-8 rounded-xl flex items-center justify-center bg-gradient-to-br shadow-lg ${isOfflineActive ? 'from-orange-500 to-red-600' : 'from-emerald-500 to-teal-600'}`}>
            <Sparkles size={16} className="text-white" />
          </div>
          <div className="flex flex-col">
              <span className="font-bold text-lg tracking-tight leading-none">Hizmet Dili</span>
              <span className="text-[9px] text-slate-400 font-medium tracking-wide mt-0.5">Developer by Ali Tellioğlu</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
           {/* View Mode Toggle */}
           <button 
             onClick={() => setViewMode(viewMode === 'chat' ? 'split' : 'chat')}
             className={`p-2 rounded-full transition-colors active:scale-95 ${viewMode === 'split' ? 'bg-blue-600 text-white' : 'bg-slate-800/50 text-slate-400 hover:bg-slate-800'}`}
           >
             {viewMode === 'chat' ? <SplitSquareVertical size={18} /> : <MessageSquare size={18} />}
           </button>

           <div className={`px-2.5 py-1 rounded-full text-[10px] font-bold border ${isOfflineActive ? 'bg-orange-500/10 border-orange-500/20 text-orange-400' : 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'}`}>
            {isOfflineActive ? 'OFFLINE' : 'ONLINE'}
           </div>
           
          <button onClick={() => setShowSettings(true)} className="p-2 rounded-full bg-slate-800/50 hover:bg-slate-800 transition-colors active:scale-95">
            <Settings size={20} className="text-slate-400" />
          </button>
        </div>
      </header>
      
      {/* Direction Indicator (Only in Chat Mode) */}
      {viewMode === 'chat' && (
      <div className="absolute top-16 left-0 right-0 z-10 flex justify-center mt-2 animate-fade-in-down pointer-events-auto">
          <button 
             onClick={toggleOfflineDirection}
             disabled={!isOfflineActive} // Only interactive in offline mode
             className={`bg-slate-800/80 backdrop-blur-md px-4 py-2 rounded-full border border-slate-700 flex items-center gap-3 text-xs font-medium text-slate-300 shadow-xl transition-all active:scale-95 ${isOfflineActive ? 'hover:bg-slate-700 cursor-pointer' : 'cursor-default'}`}
          >
              <div className="flex items-center gap-1.5">
                  <span className="text-lg">{sourceDetails.flag}</span>
                  <span className={isOfflineReverse ? "opacity-50" : "font-bold text-white"}>{sourceDetails.name}</span>
              </div>
              <ArrowRightLeft size={14} className={`text-blue-400 transition-transform duration-300 ${isOfflineReverse ? 'rotate-180' : ''}`} />
              <div className="flex items-center gap-1.5">
                  <span className={isOfflineReverse ? "font-bold text-white" : "opacity-50"}>{targetDetails.name}</span>
                  <span className="text-lg">{targetDetails.flag}</span>
              </div>
          </button>
      </div>
      )}

      {/* Main Area */}
      <main className="flex-1 flex flex-col z-10 relative overflow-hidden">
        
        {/* --- VIEW MODE: SPLIT (FACE-TO-FACE) --- */}
        {viewMode === 'split' ? (
             <div className="flex-1 flex flex-col items-stretch h-full pb-[calc(8rem+env(safe-area-inset-bottom))]">
                {/* Top Half (Target Language - Rotated 180deg) */}
                <div className="flex-1 bg-slate-900/50 border-b border-slate-800 p-6 flex flex-col justify-center items-center relative rotate-180">
                     <div className="absolute top-4 left-4 flex items-center gap-2 opacity-50">
                        <span className="text-2xl">{targetDetails.flag}</span>
                        <span className="text-sm font-bold">{targetDetails.name}</span>
                     </div>
                     <div className="text-center">
                         <div className="text-3xl font-bold text-emerald-300 leading-snug">
                            {realtimeOutput || lastModelMsg?.text || "..."}
                         </div>
                     </div>
                </div>

                {/* Bottom Half (Source Language) */}
                <div className="flex-1 bg-slate-950/50 p-6 flex flex-col justify-center items-center relative">
                     <div className="absolute top-4 left-4 flex items-center gap-2 opacity-50">
                        <span className="text-2xl">{sourceDetails.flag}</span>
                        <span className="text-sm font-bold">{sourceDetails.name}</span>
                     </div>
                     <div className="text-center">
                         <div className="text-3xl font-bold text-slate-100 leading-snug">
                            {realtimeInput || lastUserMsg?.text || "..."}
                         </div>
                     </div>
                </div>
             </div>
        ) : (
        /* --- VIEW MODE: CHAT (STANDARD) --- */
        <div ref={chatContainerRef} className="flex-1 overflow-y-auto px-4 pt-10 pb-6 space-y-6 scroll-smooth no-scrollbar">
          {error && (
            <div className="bg-red-500/10 border border-red-500/20 text-red-200 p-3 rounded-xl text-sm flex justify-between items-center mx-2">
              <span>{error}</span>
              <button onClick={() => setError(null)}><X size={16} /></button>
            </div>
          )}

          {messages.length === 0 && !realtimeInput && (
             <div className="h-full flex flex-col items-center justify-center text-slate-500 opacity-40 space-y-4">
                <Globe size={64} strokeWidth={1} />
                <p className="text-center text-sm px-10">
                  {isOfflineActive && isOfflineReverse 
                     ? `${targetDetails.name} konuşmak için mikrofona dokunun.`
                     : isOfflineActive 
                       ? `${sourceDetails.name} konuşmak için mikrofona dokunun.`
                       : "Konuşmaya başlamak için aşağıdaki mikrofona dokunun."}
                </p>
             </div>
          )}
          
          {messages.map((msg) => {
             const msgLangDetails = getLangDetails(msg.langCode);
             return (
            <div 
              key={msg.id} 
              onClick={() => setFocusedMessage(msg)}
              className={`flex flex-col cursor-pointer active:scale-[0.98] transition-transform ${msg.role === 'user' ? 'items-start' : 'items-end'}`}
            >
              <span className="text-[10px] text-slate-400 mb-1 px-1 flex items-center gap-1.5">
                  {msg.role === 'user' ? (
                      <>
                        <Mic size={10} />
                        <span>Konuşmacı</span>
                      </>
                  ) : (
                      <>
                        <Sparkles size={10} />
                        <span>Çevirmen</span>
                      </>
                  )}
              </span>
              <div className={`max-w-[85%] px-4 py-3 shadow-sm text-[15px] leading-relaxed relative group ${
                msg.role === 'user' 
                  ? 'bg-slate-800 text-slate-200 rounded-2xl rounded-tl-sm' 
                  : isOfflineActive 
                    ? 'bg-orange-600 text-white rounded-2xl rounded-tr-sm'
                    : 'bg-emerald-600 text-white rounded-2xl rounded-tr-sm'
              }`}>
                <div className="flex gap-3">
                    <span className="text-2xl shrink-0 leading-none pt-0.5 filter drop-shadow-md select-none">{msgLangDetails.flag}</span>
                    <span className="break-words">{msg.text}</span>
                </div>
                <div className="absolute right-2 bottom-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <Maximize2 size={12} className="text-white/50" />
                </div>
              </div>
            </div>
          )})}

          {/* Real-time Bubbles */}
          {(realtimeInput || realtimeOutput) && (
             <div className="space-y-4">
                {realtimeInput && (
                  <div className="flex flex-col items-start opacity-70">
                     <span className="text-[10px] text-blue-400 mb-1 px-1 animate-pulse flex items-center gap-1"><Mic size={10}/> 
                     {isOfflineActive 
                       ? (isOfflineReverse ? targetDetails.short : sourceDetails.short) 
                       : 'GİRİŞ'}...
                     </span>
                     <div className="max-w-[85%] px-5 py-3 bg-slate-800/50 border border-slate-700 border-dashed text-slate-300 rounded-2xl rounded-tl-sm">
                       {realtimeInput}
                     </div>
                  </div>
                )}
                {realtimeOutput && (
                  <div className="flex flex-col items-end opacity-70">
                     <span className="text-[10px] text-emerald-400 mb-1 px-1 animate-pulse flex items-center gap-1"><Sparkles size={10}/> ÇEVİRİYOR...</span>
                     <div className="max-w-[85%] px-5 py-3 bg-emerald-900/30 border border-emerald-500/30 border-dashed text-emerald-200 rounded-2xl rounded-tr-sm">
                       {realtimeOutput}
                     </div>
                  </div>
                )}
             </div>
          )}
        </div>
        )}

        {/* BOTTOM DOCK (Controls) */}
        {/* pb-[calc(1.5rem+env(safe-area-inset-bottom))] ensures button is above home indicator */}
        <div className="bg-slate-950/80 backdrop-blur-xl border-t border-slate-800/50 px-6 pt-4 z-30 pb-[calc(1.5rem+env(safe-area-inset-bottom,20px))] absolute bottom-0 w-full">
          
          {/* Visualizer Bar (Above Dock) */}
          <div className="h-10 w-full flex items-center justify-center mb-4">
            {isConnected ? (
              <div className="w-full h-full opacity-80">
                 {/* Updated to Green color by default if online */}
                 <AudioVisualizer analyser={inputAnalyserRef.current} isActive={true} color={isOfflineActive ? '#f97316' : '#4ade80'} />
              </div>
            ) : (
              <div className="h-1 w-16 bg-slate-800 rounded-full"></div>
            )}
          </div>

          <div className="flex items-center justify-between max-w-sm mx-auto">
            
            {/* Left: Source Language Selector */}
            <div className="w-24 flex flex-col items-center gap-1">
               <button 
                 onClick={() => setSelectorType('source')}
                 className="w-full h-16 bg-slate-800 border border-slate-700 text-slate-200 rounded-2xl flex items-center justify-center hover:border-slate-500 transition-colors active:scale-95 touch-manipulation relative overflow-hidden group"
               >
                  <span className="text-4xl filter drop-shadow-md z-10 transition-transform group-active:scale-90">{sourceDetails.flag}</span>
                  <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent"></div>
               </button>
               <span className="text-[10px] text-slate-500 font-medium">Kaynak</span>
            </div>

            {/* Center: Main Mic Toggle */}
            <div className="relative -mt-8">
              
              {/* Green Halo Ripple Effects */}
              {isConnected && (
                <>
                  <div className={`absolute inset-0 rounded-full animate-[ping_2s_cubic-bezier(0,0,0.2,1)_infinite] ${isOfflineActive ? 'bg-orange-500/40' : 'bg-green-500/40'}`}></div>
                  <div className={`absolute inset-0 rounded-full animate-[ping_2s_cubic-bezier(0,0,0.2,1)_infinite_200ms] ${isOfflineActive ? 'bg-orange-500/30' : 'bg-green-500/30'}`}></div>
                   <div className={`absolute inset-0 rounded-full animate-[ping_2s_cubic-bezier(0,0,0.2,1)_infinite_500ms] ${isOfflineActive ? 'bg-orange-500/10' : 'bg-green-500/10'}`}></div>
                </>
              )}
              
              <button
                onClick={handleMicToggle}
                className={`relative w-20 h-20 rounded-full flex items-center justify-center shadow-2xl transition-all duration-300 transform active:scale-95 touch-manipulation z-10 ${
                  isConnecting
                    ? 'bg-slate-800 border-4 border-slate-700 cursor-wait'
                    : isConnected
                      ? isOfflineActive 
                        ? 'bg-orange-600 text-white shadow-orange-600/40 ring-4 ring-orange-900/50 scale-105' 
                        : 'bg-green-600 text-white shadow-green-600/40 ring-4 ring-green-900/50 scale-105'
                      : 'bg-slate-100 text-slate-900 hover:bg-white border-4 border-slate-300 shadow-white/10'
                }`}
              >
                {isConnecting ? (
                   <Loader2 size={32} className="animate-spin text-slate-400" />
                ) : isConnected ? (
                   <Square size={28} fill="currentColor" className="rounded-sm" />
                ) : (
                   <Mic size={32} strokeWidth={2.5} />
                )}
              </button>
            </div>

            {/* Right: Target Language Selector */}
            <div className="w-24 flex flex-col items-center gap-1">
               <button 
                 onClick={() => setSelectorType('target')}
                 className="w-full h-16 bg-slate-800 border border-slate-700 text-slate-200 rounded-2xl flex items-center justify-center hover:border-slate-500 transition-colors active:scale-95 touch-manipulation relative overflow-hidden group"
               >
                  <span className="text-4xl filter drop-shadow-md z-10 transition-transform group-active:scale-90">{targetDetails.flag}</span>
                  <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent"></div>
               </button>
               <span className="text-[10px] text-slate-500 font-medium">Hedef</span>
            </div>

          </div>
        </div>

      </main>

      {/* FULLSCREEN FOCUSED MESSAGE MODAL */}
      {focusedMessage && (
          <div 
             className="fixed inset-0 z-50 bg-black/95 backdrop-blur-xl flex flex-col items-center justify-center p-8 animate-fade-in"
             onClick={() => setFocusedMessage(null)}
          >
              <button className="absolute top-6 right-6 p-4 rounded-full bg-slate-800 text-white hover:bg-slate-700">
                  <Minimize2 size={24} />
              </button>
              
              <div className="text-center space-y-6">
                  <div className="text-6xl animate-bounce">
                    {getLangDetails(focusedMessage.langCode).flag}
                  </div>
                  <p className={`text-4xl font-bold leading-tight ${
                      focusedMessage.role === 'model' 
                      ? isOfflineActive ? 'text-orange-400' : 'text-emerald-400' 
                      : 'text-white'
                  }`}>
                      {focusedMessage.text}
                  </p>
                  <p className="text-slate-500 text-lg uppercase tracking-widest mt-4">
                      {focusedMessage.role === 'user' ? 'Konuşmacı' : 'Çeviri'}
                  </p>
              </div>
          </div>
      )}

      {/* Language Selector Sheet */}
      {selectorType && (
          <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/60 backdrop-blur-sm animate-fade-in" onClick={() => setSelectorType(null)}>
              <div className="bg-slate-900 border-t border-slate-700 w-full max-w-lg rounded-t-3xl shadow-2xl overflow-hidden mb-[env(safe-area-inset-bottom)]" onClick={e => e.stopPropagation()}>
                  <div className="p-4 flex items-center justify-center relative border-b border-slate-800">
                      <div className="w-12 h-1 bg-slate-700 rounded-full absolute top-3"></div>
                      <h3 className="text-white font-semibold mt-2 flex items-center gap-2">
                        <Globe size={18} className="text-blue-400" />
                        {selectorType === 'source' ? 'Kaynak Dil Seçin' : 'Hedef Dil Seçin'}
                      </h3>
                  </div>
                  <div className="p-6 grid grid-cols-2 gap-3 pb-8">
                      {LANGUAGE_META.map((lang) => {
                          const isSelected = (selectorType === 'source' && sourceLang === lang.code) || 
                                             (selectorType === 'target' && targetLang === lang.code);
                          return (
                          <button
                              key={lang.code}
                              onClick={() => handleLanguageSelect(lang.code)}
                              className={`p-4 rounded-xl border flex flex-col items-center gap-2 transition-all active:scale-95 touch-manipulation ${
                                  isSelected 
                                  ? 'bg-blue-600/20 border-blue-500 text-white shadow-lg shadow-blue-900/20' 
                                  : 'bg-slate-800 border-slate-700 text-slate-400 hover:bg-slate-750 hover:border-slate-600'
                              }`}
                          >
                              <span className="text-4xl filter drop-shadow-md">{lang.flag}</span>
                              <span className="font-medium text-sm">{lang.name}</span>
                          </button>
                      )})}
                  </div>
              </div>
          </div>
      )}

      {/* Settings Sheet */}
      {showSettings && (
        <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/60 backdrop-blur-sm animate-fade-in" onClick={() => setShowSettings(false)}>
          <div className="bg-slate-900 border-t border-slate-700 w-full max-w-lg rounded-t-3xl shadow-2xl overflow-hidden max-h-[85vh] flex flex-col mb-[env(safe-area-inset-bottom)]" onClick={e => e.stopPropagation()}>
             <div className="p-5 flex justify-between items-center bg-slate-800/50">
                <h2 className="text-lg font-bold flex items-center gap-2 text-white">
                  <Settings size={20} className="text-blue-400"/> Ayarlar
                </h2>
                <div className="w-12 h-1 bg-slate-700 rounded-full absolute top-3 left-1/2 -translate-x-1/2"></div>
                <button onClick={() => setShowSettings(false)} className="p-1 bg-slate-800 rounded-full text-slate-400"><X size={20} /></button>
             </div>
             
             <div className="p-6 overflow-y-auto space-y-6 pb-10">
               {/* Mode */}
               <div className="bg-slate-800/30 p-4 rounded-2xl border border-slate-800">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-semibold text-slate-200">Çevrimdışı Mod</span>
                    <button onClick={() => { if(isConnected) stopConnection(); setForceOfflineMode(!forceOfflineMode); }} 
                      className={`w-12 h-7 rounded-full p-1 transition-colors ${forceOfflineMode ? 'bg-orange-500' : 'bg-slate-700'}`}>
                      <div className={`w-5 h-5 rounded-full bg-white transition-transform ${forceOfflineMode ? 'translate-x-5' : ''}`}></div>
                    </button>
                  </div>
                  <p className="text-xs text-slate-500">İnternet bağlantısı olmadan cihaz üzerinde basit çeviri yapın.</p>
               </div>

               {/* Voice Selection */}
               <div className="bg-slate-800/30 p-4 rounded-2xl border border-slate-800">
                   <div className="flex items-center gap-2 mb-3">
                       <User size={16} className="text-blue-400" />
                       <span className="font-semibold text-slate-200 text-sm">Ses Tercihi</span>
                   </div>
                   <div className="grid grid-cols-2 gap-3">
                       <button 
                         onClick={() => handleVoiceChange('female')}
                         className={`py-3 px-4 rounded-xl flex items-center justify-center gap-2 transition-all ${voiceType === 'female' ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/30' : 'bg-slate-800 text-slate-400 border border-slate-700'}`}
                       >
                           <span className="text-lg">👩</span>
                           <span className="font-medium text-sm">Kadın</span>
                       </button>
                       <button 
                         onClick={() => handleVoiceChange('male')}
                         className={`py-3 px-4 rounded-xl flex items-center justify-center gap-2 transition-all ${voiceType === 'male' ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/30' : 'bg-slate-800 text-slate-400 border border-slate-700'}`}
                       >
                           <span className="text-lg">👨</span>
                           <span className="font-medium text-sm">Erkek</span>
                       </button>
                   </div>
                   <p className="text-xs text-slate-500 mt-2">Çevirmen sesi (Sadece Online modda geçerlidir).</p>
               </div>

               {/* Packs */}
               <div>
                 <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3 px-1">Dil Paketleri</h3>
                 <div className="space-y-3">
                    {offlinePacks.map(pack => {
                       // Find flag based on pack ID content (simple heuristic for demo)
                       let flag = '🏳️';
                       if (pack.id.includes('en')) flag = '🇬🇧';
                       if (pack.id.includes('nl')) flag = '🇳🇱';
                       if (pack.id.includes('ar')) flag = '🇸🇦';
                       if (pack.id.includes('ru')) flag = '🇷🇺';
                       if (pack.id.includes('de')) flag = '🇩🇪';
                       if (pack.id.includes('it')) flag = '🇮🇹';
                       if (pack.id.includes('fr')) flag = '🇫🇷';
                       
                       return (
                       <div key={pack.id} className="bg-slate-800/50 p-3 rounded-xl border border-slate-700/50 flex items-center justify-between">
                          <div className="flex-1">
                             <div className="flex items-center gap-2">
                               <span className="text-lg">{flag}</span>
                               <span className="font-semibold text-sm">{pack.pair}</span>
                               <span className="text-[10px] bg-slate-700 text-slate-300 px-1.5 py-0.5 rounded">{pack.size}</span>
                             </div>
                             <div className="text-xs text-slate-400 mt-1">{pack.name}</div>
                             {pack.progress > 0 && pack.progress < 100 && (
                               <div className="w-full h-1 bg-slate-700 mt-2 rounded-full overflow-hidden">
                                  <div className="h-full bg-blue-500 transition-all" style={{width: `${pack.progress}%`}}></div>
                               </div>
                             )}
                          </div>
                          
                          <div className="ml-4">
                             {pack.downloaded ? (
                               <button onClick={() => deletePack(pack.id)} className="p-2 text-green-400 bg-green-400/10 rounded-full hover:bg-red-500/20 hover:text-red-400 transition-colors">
                                  <Check size={18} />
                               </button>
                             ) : pack.progress > 0 ? (
                               <span className="text-xs font-bold text-blue-400">{pack.progress}%</span>
                             ) : (
                               <button onClick={() => downloadPack(pack.id)} className="p-2 bg-slate-700 hover:bg-blue-600 rounded-full text-white transition-colors">
                                 <Download size={18} />
                               </button>
                             )}
                          </div>
                       </div>
                    )})}
                 </div>
               </div>
               
               {/* Clear History */}
               <button onClick={() => { setMessages([]); setRealtimeInput(''); setRealtimeOutput(''); setShowSettings(false); }} className="w-full py-4 text-sm text-red-400 font-medium bg-red-500/10 rounded-2xl hover:bg-red-500/20 transition-colors flex items-center justify-center gap-2 active:scale-95 touch-manipulation">
                  <Trash2 size={16} /> Geçmişi Temizle
               </button>
             </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;