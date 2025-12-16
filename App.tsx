import React, { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality } from '@google/genai';
import { TargetLanguage, ChatMessage, OfflinePack, ArchivedSession } from './types';
import { float32To16BitPCM, arrayBufferToBase64, base64ToArrayBuffer, pcm16ToFloat32 } from './utils/audioUtils';
import AudioVisualizer from './components/AudioVisualizer';
import { Mic, Globe, Settings, RotateCcw, Wifi, WifiOff, Download, Check, Trash2, X, Zap, Square, ChevronDown, Sparkles, Loader2, Languages, ArrowRightLeft, ArrowRight, User, SplitSquareVertical, Maximize2, Minimize2, MessageSquare, Ear, ScrollText, Save, FolderOpen, Calendar, ChevronRight, FileText, Waves, Key, LogOut, ExternalLink } from 'lucide-react';

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
  { code: TargetLanguage.UKRAINIAN, name: 'Ukraynaca', flag: '🇺🇦', short: 'UA', locale: 'uk-UA' },
];

const getLangDetails = (lang?: TargetLanguage) => {
    if (!lang) return LANGUAGE_META[0];
    return LANGUAGE_META.find(l => l.code === lang) || LANGUAGE_META[0];
};

// --- ROBUST CLIENT-SIDE LANGUAGE DETECTION ---
const COMMON_WORDS: Record<string, Set<string>> = {
  [TargetLanguage.TURKISH]: new Set([
    've', 'bir', 'bu', 'da', 'de', 'için', 'ben', 'sen', 'o', 'biz', 'siz', 'onlar', 
    'ama', 'fakat', 'ile', 'ne', 'gibi', 'var', 'yok', 'çok', 'daha', 'en', 'kadar', 
    'olarak', 'diye', 'zaman', 'şey', 'bunu', 'şunu', 'bana', 'sana', 'ona', 'evet', 
    'hayır', 'merhaba', 'nasıl', 'neden', 'niçin', 'kim', 'mu', 'mı', 'mi', 'mü'
  ]),
  [TargetLanguage.ENGLISH]: new Set([
    'the', 'and', 'is', 'it', 'to', 'in', 'you', 'that', 'of', 'for', 'on', 'are', 
    'with', 'as', 'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had', 'by', 
    'word', 'but', 'not', 'what', 'all', 'were', 'we', 'when', 'your', 'can', 'said', 
    'there', 'use', 'an', 'each', 'which', 'she', 'do', 'how', 'their', 'if', 'hello', 'hi'
  ]),
  [TargetLanguage.DUTCH]: new Set([
    'de', 'het', 'een', 'en', 'van', 'ik', 'te', 'dat', 'die', 'in', 'is', 'op', 
    'voor', 'met', 'niet', 'zijn', 'er', 'wat', 'maar', 'om', 'ook', 'als', 'bij', 
    'of', 'uw', 'je', 'hij', 'u', 'aan', 'zo', 'dan', 'hij', 'wij', 'we', 'ze', 
    'hallo', 'ja', 'nee', 'goed', 'waar', 'waarom', 'hoe'
  ]),
  [TargetLanguage.GERMAN]: new Set([
    'die', 'der', 'und', 'in', 'zu', 'den', 'das', 'nicht', 'von', 'sie', 'ist', 
    'des', 'sich', 'mit', 'dem', 'dass', 'er', 'es', 'ein', 'ich', 'auf', 'so', 
    'eine', 'auch', 'als', 'an', 'nach', 'wie', 'im', 'für', 'man', 'aber', 'aus', 
    'hallo', 'ja', 'nein', 'wir', 'ihr', 'warum', 'wer'
  ]),
  [TargetLanguage.FRENCH]: new Set([
    'le', 'la', 'les', 'de', 'et', 'un', 'une', 'est', 'je', 'à', 'en', 'que', 
    'du', 'il', 'elle', 'dans', 'pour', 'pas', 'sur', 'au', 'ce', 'ne', 'plus', 
    'se', 'par', 'avec', 'tout', 'faire', 'son', 'ses', 'sa', 'mais', 'nous', 
    'vous', 'ils', 'elles', 'bonjour', 'oui', 'non', 'merci'
  ]),
  [TargetLanguage.ITALIAN]: new Set([
    'il', 'la', 'i', 'gli', 'le', 'di', 'e', 'che', 'un', 'una', 'è', 'in', 
    'per', 'non', 'con', 'sono', 'ma', 'come', 'questo', 'quello', 'più', 'o', 
    'io', 'tu', 'lui', 'lei', 'noi', 'voi', 'loro', 'ciao', 'si', 'no', 'perché', 
    'dove', 'quando', 'chi'
  ]),
  [TargetLanguage.UKRAINIAN]: new Set([
    'і', 'в', 'на', 'не', 'що', 'з', 'я', 'як', 'це', 'до', 'ми', 'ти', 'він', 'вона', 'вони', 
    'але', 'так', 'ні', 'привіт', 'дякую', 'будь', 'ласка', 'добрий', 'день', 'мене', 'звати',
    'дуже', 'добре', 'де', 'коли', 'хто', 'чому', 'ви', 'вас', 'мені', 'нам', 'їх'
  ])
};

const detectLanguage = (text: string, langA: TargetLanguage, langB: TargetLanguage): TargetLanguage => {
  if (!text || text.length < 2) return langA; 

  const t = text.toLowerCase().trim();

  const isArabic = /[\u0600-\u06FF]/.test(text);
  const isCyrillic = /[\u0400-\u04FF]/.test(text);

  if (langA === TargetLanguage.ARABIC && isArabic) return langA;
  if (langB === TargetLanguage.ARABIC && isArabic) return langB;
  
  if (isCyrillic) {
      const isUkrainianSpecific = /[іїєґІЇЄҐ]/.test(text);
      const isRussianSpecific = /[ыэёъЫЭЁЪ]/.test(text);

      if (isUkrainianSpecific) {
           if (langA === TargetLanguage.UKRAINIAN) return langA;
           if (langB === TargetLanguage.UKRAINIAN) return langB;
      }
      if (isRussianSpecific) {
           if (langA === TargetLanguage.RUSSIAN) return langA;
           if (langB === TargetLanguage.RUSSIAN) return langB;
      }
      const isACyrillicLang = langA === TargetLanguage.RUSSIAN || langA === TargetLanguage.UKRAINIAN;
      const isBCyrillicLang = langB === TargetLanguage.RUSSIAN || langB === TargetLanguage.UKRAINIAN;
      if (isACyrillicLang && !isBCyrillicLang) return langA;
      if (!isACyrillicLang && isBCyrillicLang) return langB;
  }

  if (/[ğıİşŞ]/.test(text)) {
      if (langA === TargetLanguage.TURKISH) return langA;
      if (langB === TargetLanguage.TURKISH) return langB;
  }
  if (/ß/.test(text)) {
      if (langA === TargetLanguage.GERMAN) return langA;
      if (langB === TargetLanguage.GERMAN) return langB;
  }
  if (/œ/.test(text)) {
      if (langA === TargetLanguage.FRENCH) return langA;
      if (langB === TargetLanguage.FRENCH) return langB;
  }

  const getScore = (l: TargetLanguage) => {
      if (!COMMON_WORDS[l]) return 0;
      const words = t.split(/[\s,.?!:;"']+/);
      let matchCount = 0;
      for (const w of words) {
          if (w.length < 2) continue;
          if (COMMON_WORDS[l].has(w)) matchCount++;
      }
      return matchCount;
  };

  const scoreA = getScore(langA);
  const scoreB = getScore(langB);

  if (scoreA > scoreB) return langA;
  if (scoreB > scoreA) return langB;

  return langA;
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
  { id: 'tr-ua', pair: 'TR ↔ UA', name: 'Ukraynaca', size: '47 MB', downloaded: false, progress: 0 },
];

const App: React.FC = () => {
  const [sourceLang, setSourceLang] = useState<TargetLanguage>(TargetLanguage.TURKISH);
  const [targetLang, setTargetLang] = useState<TargetLanguage>(TargetLanguage.ENGLISH);
  const [voiceType, setVoiceType] = useState<'female' | 'male'>('female');
  const [selectorType, setSelectorType] = useState<'source' | 'target' | null>(null);

  // API Key State
  const [apiKey, setApiKey] = useState<string>('');
  const [tempApiKeyInput, setTempApiKeyInput] = useState('');

  // View Modes: 'chat' | 'split' | 'listen' | 'archive'
  const [viewMode, setViewMode] = useState<'chat' | 'split' | 'listen' | 'archive'>('chat');
  const [focusedMessage, setFocusedMessage] = useState<ChatMessage | null>(null);
  const [openedArchive, setOpenedArchive] = useState<ArchivedSession | null>(null);

  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  
  // Listen Mode specific state
  const [isListenModeActive, setIsListenModeActive] = useState(false);
  const [showSaveModal, setShowSaveModal] = useState(false);

  const [isOfflineReverse, setIsOfflineReverse] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [savedSessions, setSavedSessions] = useState<ArchivedSession[]>([]);
  const [error, setError] = useState<string | null>(null);
  
  const [realtimeInput, setRealtimeInput] = useState('');
  const [realtimeOutput, setRealtimeOutput] = useState('');
  
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [forceOfflineMode, setForceOfflineMode] = useState(false);
  // Noise Mode (Push-to-Talk)
  const [isNoiseMode, setIsNoiseMode] = useState(false);
  const [isHoldingMic, setIsHoldingMic] = useState(false);

  const [showSettings, setShowSettings] = useState(false);
  const [offlinePacks, setOfflinePacks] = useState<OfflinePack[]>(INITIAL_PACKS);
  
  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const inputAnalyserRef = useRef<AnalyserNode | null>(null);
  const outputAnalyserRef = useRef<AnalyserNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  
  const aiClientRef = useRef<GoogleGenAI | null>(null);
  const sessionPromiseRef = useRef<Promise<any> | null>(null);
  const activeSessionRef = useRef<any>(null);
  const shouldReconnectRef = useRef(false);
  const nextStartTimeRef = useRef<number>(0);
  const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const isTransmittingRef = useRef(false);
  
  const currentInputTranscription = useRef('');
  const currentOutputTranscription = useRef('');
  const recognitionRef = useRef<any>(null);

  // Smart Scrolling Ref
  const isUserScrolledUpRef = useRef(false);

  useEffect(() => {
    // Check Local Storage for API Key
    const storedKey = localStorage.getItem('gemini_api_key');
    if (storedKey) {
        setApiKey(storedKey);
        aiClientRef.current = new GoogleGenAI({ apiKey: storedKey });
    } else {
        // Safe check for process.env (prevents crash in browser)
        const envKey = (typeof process !== 'undefined' && process.env) ? process.env.API_KEY : undefined;
        if (envKey) {
            setApiKey(envKey);
            aiClientRef.current = new GoogleGenAI({ apiKey: envKey });
        } else {
            setApiKey(''); 
        }
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
    const savedVoice = localStorage.getItem('voiceType');
    if (savedVoice === 'male' || savedVoice === 'female') {
        setVoiceType(savedVoice);
    }
    
    // Load Noise Mode Preference
    const savedNoiseMode = localStorage.getItem('isNoiseMode');
    if (savedNoiseMode) {
        setIsNoiseMode(JSON.parse(savedNoiseMode));
    }

    const loadedSessions = localStorage.getItem('archivedSessions');
    if (loadedSessions) {
        setSavedSessions(JSON.parse(loadedSessions));
    }

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  // Screen Wake Lock
  useEffect(() => {
    let wakeLock: any = null;
    const requestWakeLock = async () => {
      if ('wakeLock' in navigator && isConnected) {
        try {
          wakeLock = await (navigator as any).wakeLock.request('screen');
        } catch (err) {
          console.log('Wake Lock Error:', err);
        }
      }
    };
    if (isConnected) requestWakeLock();
    return () => {
      if (wakeLock) wakeLock.release();
    };
  }, [isConnected]);

  const isOfflineActive = !isOnline || forceOfflineMode;

  const handleSaveApiKey = () => {
      if (!tempApiKeyInput.trim()) return;
      const key = tempApiKeyInput.trim();
      localStorage.setItem('gemini_api_key', key);
      setApiKey(key);
      aiClientRef.current = new GoogleGenAI({ apiKey: key });
      setTempApiKeyInput('');
      setError(null);
  };

  const handleDeleteApiKey = () => {
      localStorage.removeItem('gemini_api_key');
      setApiKey('');
      aiClientRef.current = null;
      setShowSettings(false);
      stopConnection();
  };

  const triggerHaptic = () => {
      if (navigator.vibrate) navigator.vibrate(50);
  };

  const processAudioInput = useCallback((inputData: Float32Array) => {
    if (!sessionPromiseRef.current) return;
    if (!isTransmittingRef.current) return; 

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

  const stopConnection = useCallback(() => {
    setIsConnecting(false);
    setIsConnected(false);
    setIsListenModeActive(false); 
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

  const startLiveSession = async (mode: 'bidirectional' | 'listen') => {
      triggerHaptic();
      
      if (isConnecting) return;
      if (isConnected) {
         if (!isNoiseMode) {
            if (isListenModeActive) {
                if (messages.length > 0) {
                    isTransmittingRef.current = false;
                    setShowSaveModal(true);
                } else {
                    stopConnection();
                }
            } else {
                stopConnection();
            }
         }
         return;
      }

      setIsConnecting(true);
      setError(null);
      
      const isListen = mode === 'listen';
      if (isListen) {
          setViewMode('listen');
          setIsListenModeActive(true);
          setMessages([]); 
      }

      if (isOfflineActive) {
          setTimeout(() => {
            setIsConnected(true);
            isTransmittingRef.current = true;
            setIsConnecting(false);
            startOfflineRecognition();
          }, 300);
          return;
      }

      // Check if client is ready (Key exists)
      if (!aiClientRef.current) {
           // Double check store
           const stored = localStorage.getItem('gemini_api_key');
           if (stored) {
               aiClientRef.current = new GoogleGenAI({ apiKey: stored });
           } else {
               setError("API Anahtarı eksik. Lütfen ayarlardan giriniz.");
               setIsConnecting(false);
               setIsListenModeActive(false);
               return;
           }
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

        const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: INPUT_SAMPLE_RATE });
        inputAudioContextRef.current = audioCtx;
        const source = audioCtx.createMediaStreamSource(stream);
        
        let nodeToConnectToAnalyser = source;

        if (isNoiseMode) {
            const highPassFilter = audioCtx.createBiquadFilter();
            highPassFilter.type = 'highpass';
            highPassFilter.frequency.value = 100; 
            highPassFilter.Q.value = 0.5;
            source.connect(highPassFilter);
            nodeToConnectToAnalyser = highPassFilter as any;
        }

        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 256;
        inputAnalyserRef.current = analyser;
        
        const scriptProcessor = audioCtx.createScriptProcessor(4096, 1, 1);
        scriptProcessorRef.current = scriptProcessor;
        scriptProcessor.onaudioprocess = (e) => {
          const inputData = e.inputBuffer.getChannelData(0);
          processAudioInput(inputData);
        };

        nodeToConnectToAnalyser.connect(analyser);
        analyser.connect(scriptProcessor);
        scriptProcessor.connect(audioCtx.destination);

        const outCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: OUTPUT_SAMPLE_RATE });
        outputAudioContextRef.current = outCtx;
        const outAnalyser = outCtx.createAnalyser();
        outAnalyser.fftSize = 256;
        outputAnalyserRef.current = outAnalyser;
        outAnalyser.connect(outCtx.destination);

        const sourceDetails = getLangDetails(sourceLang);
        const targetDetails = getLangDetails(targetLang);
        const voiceName = voiceType === 'female' ? 'Kore' : 'Fenrir';

        let systemPrompt = "";
        const modalities: Modality[] = [Modality.AUDIO]; 

        if (isListen) {
            systemPrompt = `You are a professional simultaneous interpreter.
Source Language: ${targetDetails.name}
Target Language: ${sourceDetails.name} (Turkish)
TASK: Translate the dominant speaker's speech into Turkish IMMEDIATELY.
CRITICAL PROTOCOLS:
1. **FOCUS ON DOMINANT VOICE:** Ignore background noise.
2. **STREAMING OUTPUT:** Translate phrase-by-phrase.
3. **PUNCTUATION:** Use correct punctuation (., ?, !) to clearly mark the end of segments.`;
        } else {
            systemPrompt = `You are a professional bidirectional interpreter.
Languages: ${sourceDetails.name} <-> ${targetDetails.name}.
CORE FUNCTION: Listen, detect language, translate to the OTHER language.
Focus on the dominant voice. Ignore background noise.`;
        }

        const connectPromise = aiClientRef.current.live.connect({
          model: MODEL_NAME,
          config: {
            systemInstruction: systemPrompt,
            responseModalities: modalities, 
            inputAudioTranscription: {},
            outputAudioTranscription: {}, 
            speechConfig: {
              voiceConfig: { prebuiltVoiceConfig: { voiceName: voiceName } }
            }
          },
          callbacks: {
            onopen: () => {
              console.log("Connected");
              setIsConnecting(false);
              setIsConnected(true);
              isTransmittingRef.current = isNoiseMode ? false : true; 
              nextStartTimeRef.current = 0;
              setRealtimeInput('');
              setRealtimeOutput('');
            },
            onmessage: (msg: LiveServerMessage) => handleServerMessage(msg, isListen),
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

  // --- NOISE MODE HANDLERS (Push-to-Talk) ---
  const handlePTTStart = (mode: 'bidirectional' | 'listen') => {
      if (!isConnected) {
          startLiveSession(mode);
          setIsHoldingMic(true);
      } else {
          setIsHoldingMic(true);
          isTransmittingRef.current = true;
          triggerHaptic();
      }
  };

  const handlePTTEnd = () => {
      setIsHoldingMic(false);
      if (isConnected) {
          isTransmittingRef.current = false;
      }
  };

  const handleMicToggle = () => {
      if (isNoiseMode) return;
      startLiveSession('bidirectional');
  };
  
  const handleListenToggle = () => {
      if (isNoiseMode) return;
      startLiveSession('listen');
  };

  const saveSession = () => {
      const newSession: ArchivedSession = {
          id: Date.now().toString(),
          date: new Date().toISOString(),
          targetLang: targetLang,
          preview: messages.length > 0 ? messages[0].text.substring(0, 50) + "..." : "Boş Oturum",
          messages: [...messages]
      };
      const updatedSessions = [newSession, ...savedSessions];
      setSavedSessions(updatedSessions);
      localStorage.setItem('archivedSessions', JSON.stringify(updatedSessions));
      setMessages([]);
      setShowSaveModal(false);
      stopConnection();
  };

  const discardSession = () => {
      setMessages([]);
      setShowSaveModal(false);
      stopConnection();
  };

  const handleServerMessage = (msg: LiveServerMessage, isListen: boolean) => {
    if (msg.serverContent?.inputTranscription) {
       currentInputTranscription.current += msg.serverContent.inputTranscription.text;
       setRealtimeInput(currentInputTranscription.current);
    }

    if (msg.serverContent?.outputTranscription) {
        currentOutputTranscription.current += msg.serverContent.outputTranscription.text;
        
        if (isListen) {
          const rawText = currentOutputTranscription.current;
          const splitMatch = rawText.match(/([.?!])\s+/);

          if (splitMatch && splitMatch.index !== undefined) {
             const splitIndex = splitMatch.index + 1; 
             const sentence = rawText.substring(0, splitIndex).trim();
             const remainder = rawText.substring(splitIndex).trim(); 

             if (sentence.length > 0) {
                 addMessage('model', sentence, sourceLang);
                 currentOutputTranscription.current = remainder;
             }
          }
          setRealtimeOutput(currentOutputTranscription.current);

        } else {
          setRealtimeOutput(currentOutputTranscription.current);
        }
    }

    const audioData = msg.serverContent?.modelTurn?.parts?.find(p => p.inlineData)?.inlineData?.data;
    if (audioData && !isListen) {
        playAudioResponse(audioData);
    }

    if (msg.serverContent?.turnComplete) {
       const inputTx = currentInputTranscription.current.trim();
       const outputTx = currentOutputTranscription.current.trim();
       
       if (inputTx || outputTx) {
         if (isListen) {
             if (outputTx) {
                 addMessage('model', outputTx, sourceLang); 
             }
             setRealtimeInput(''); 
         } else {
             const detectedInputLang = detectLanguage(inputTx, sourceLang, targetLang);
             const detectedOutputLang = detectedInputLang === sourceLang ? targetLang : sourceLang;
             addMessage('user', inputTx || '...', detectedInputLang);
             addMessage('model', outputTx || '...', detectedOutputLang);
         }
       }
       
       currentInputTranscription.current = '';
       currentOutputTranscription.current = '';
       setRealtimeOutput('');
       
       if (!isListen) setRealtimeInput('');
    }
  };

  const handleLanguageSelect = (newLang: TargetLanguage) => {
      if (selectorType === 'source') {
          if (newLang === targetLang) setTargetLang(sourceLang);
          setSourceLang(newLang);
      } else if (selectorType === 'target') {
          if (newLang === sourceLang) setSourceLang(targetLang);
          setTargetLang(newLang);
      }
      setSelectorType(null);
      if (isConnected && !isOfflineActive) {
          shouldReconnectRef.current = true;
          stopConnection();
      }
  };

  const handleVoiceChange = (type: 'female' | 'male') => {
      setVoiceType(type);
      localStorage.setItem('voiceType', type);
      if (isConnected && !isOfflineActive) {
          shouldReconnectRef.current = true;
          stopConnection();
      }
  };

  const toggleNoiseMode = () => {
      const newValue = !isNoiseMode;
      setIsNoiseMode(newValue);
      localStorage.setItem('isNoiseMode', JSON.stringify(newValue));
      if (isConnected) {
          stopConnection(); 
      }
  };

  useEffect(() => {
      if (shouldReconnectRef.current) {
          shouldReconnectRef.current = false;
          handleMicToggle();
      }
  }, [sourceLang, targetLang, voiceType]);

  const toggleOfflineDirection = () => {
    if (!isOfflineActive) return;
    if (isConnected) {
        stopConnection();
        setTimeout(() => {
           setIsOfflineReverse(!isOfflineReverse);
           handleMicToggle();
        }, 200);
    } else {
        setIsOfflineReverse(!isOfflineReverse);
    }
  };

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
      const activeLocale = isOfflineReverse ? targetDetails.locale : sourceDetails.locale;
      recognition.lang = activeLocale;
      recognition.continuous = true; 
      recognition.interimResults = true; 
      recognition.onstart = () => setRealtimeInput('');
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
      }
  };

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

  const speakOffline = (text: string, locale: string) => {
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text); 
      utterance.lang = locale;
      window.speechSynthesis.speak(utterance);
    }
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

  // --- SMART SCROLL HANDLING ---
  const chatContainerRef = useRef<HTMLDivElement>(null);
  
  const handleScroll = () => {
      if (!chatContainerRef.current) return;
      const { scrollTop, scrollHeight, clientHeight } = chatContainerRef.current;
      const isAtBottom = scrollHeight - scrollTop - clientHeight < 100;
      isUserScrolledUpRef.current = !isAtBottom;
  };

  useEffect(() => {
    if (chatContainerRef.current) {
        if (!isUserScrolledUpRef.current) {
            chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
        }
    }
  }, [messages, realtimeInput, realtimeOutput, viewMode]);

  // ... (downloadPack & deletePack functions remain same)
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

  const lastUserMsg = messages.slice().reverse().find(m => m.role === 'user');
  const lastModelMsg = messages.slice().reverse().find(m => m.role === 'model');

  const sourceDetails = getLangDetails(sourceLang);
  const targetDetails = getLangDetails(targetLang);

  const detectedRealtimeInputLang = detectLanguage(realtimeInput, sourceLang, targetLang);
  const isInputTarget = detectedRealtimeInputLang === targetLang;

  const deleteArchivedSession = (id: string) => {
      const updated = savedSessions.filter(s => s.id !== id);
      setSavedSessions(updated);
      localStorage.setItem('archivedSessions', JSON.stringify(updated));
      if (openedArchive?.id === id) setOpenedArchive(null);
  };

  return (
    <div className={`h-[100dvh] w-full flex flex-col font-sans relative overflow-hidden select-none transition-colors duration-500 ${isListenModeActive ? 'bg-orange-950 text-orange-50' : 'bg-slate-950 text-slate-100'}`}>
      
      {/* Dynamic Background */}
      <div className="absolute inset-0 z-0 pointer-events-none">
        <div className={`absolute top-[-10%] left-[-20%] w-[150vw] h-[60vh] rounded-full blur-[80px] opacity-20 transition-colors duration-1000 ${isListenModeActive ? 'bg-orange-600' : isOfflineActive ? 'bg-red-600' : isConnected ? 'bg-emerald-900' : 'bg-blue-900'}`}></div>
        <div className="absolute bottom-[-10%] right-[-20%] w-[150vw] h-[50vh] bg-purple-900/10 rounded-full blur-[100px]"></div>
      </div>

      {/* API KEY MODAL (If no key is present) */}
      {!apiKey && (
          <div className="fixed inset-0 z-[100] bg-slate-950 flex flex-col items-center justify-center p-6 animate-fade-in">
             <div className="w-full max-w-sm">
                <div className="flex justify-center mb-8">
                    <div className="w-20 h-20 bg-blue-500/10 rounded-3xl flex items-center justify-center shadow-2xl shadow-blue-500/20">
                        <Key size={40} className="text-blue-400" />
                    </div>
                </div>
                <h1 className="text-2xl font-bold text-center text-white mb-2">Hoş Geldiniz</h1>
                <p className="text-slate-400 text-center mb-8">Uygulamayı kullanmak için lütfen Google Gemini API anahtarınızı giriniz.</p>
                
                <div className="space-y-4">
                    <div className="relative">
                        <div className="absolute left-4 top-3.5 text-slate-500">
                            <Key size={18} />
                        </div>
                        <input 
                           type="password"
                           value={tempApiKeyInput}
                           onChange={(e) => setTempApiKeyInput(e.target.value)}
                           placeholder="API Anahtarınızı yapıştırın"
                           className="w-full bg-slate-900 border border-slate-700 rounded-xl py-3.5 pl-12 pr-4 text-white focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all placeholder:text-slate-600"
                        />
                    </div>
                    
                    <button 
                        onClick={handleSaveApiKey}
                        disabled={!tempApiKeyInput.trim()}
                        className="w-full py-3.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-bold rounded-xl shadow-lg shadow-blue-900/30 transition-all active:scale-[0.98]"
                    >
                        Giriş Yap
                    </button>

                    <a href="https://aistudio.google.com/app/apikey" target="_blank" rel="noreferrer" className="flex items-center justify-center gap-2 text-sm text-slate-500 hover:text-blue-400 transition-colors mt-4">
                        Anahtarım yok, hemen oluştur <ExternalLink size={14} />
                    </a>
                </div>
             </div>
          </div>
      )}

      {/* Header */}
      <header className="z-20 h-16 flex items-center justify-between px-5 bg-gradient-to-b from-black/80 to-transparent backdrop-blur-sm sticky top-0 pt-[env(safe-area-inset-top)]">
        <div className="flex items-center gap-2.5">
          <div className={`w-8 h-8 rounded-xl flex items-center justify-center bg-gradient-to-br shadow-lg transition-colors duration-500 ${isListenModeActive ? 'from-orange-500 to-amber-600' : isOfflineActive ? 'from-orange-500 to-red-600' : 'from-emerald-500 to-teal-600'}`}>
            {isListenModeActive ? <Ear size={16} className="text-white" /> : <Sparkles size={16} className="text-white" />}
          </div>
          <div className="flex flex-col">
              <span className={`font-bold text-lg tracking-tight leading-none ${isListenModeActive ? 'text-orange-100' : ''}`}>
                 {viewMode === 'archive' ? 'Arşiv' : isListenModeActive ? 'Dinleme Modu' : 'Hizmet Dili'}
              </span>
              <span className={`text-[9px] font-medium tracking-wide mt-0.5 ${isListenModeActive ? 'text-orange-300' : 'text-slate-400'}`}>
                 {viewMode === 'archive' ? 'Kayıtlı Oturumlar' : isListenModeActive ? `${targetDetails.name} Dinleniyor...` : isNoiseMode ? 'Bas-Konuş Modu Aktif' : 'Developer by Ali Tellioğlu'}
              </span>
          </div>
        </div>
        <div className="flex items-center gap-2">
           {!isListenModeActive && viewMode !== 'archive' && (
             <button 
               onClick={() => setViewMode(viewMode === 'chat' ? 'split' : 'chat')}
               className={`p-2 rounded-full transition-colors active:scale-95 ${viewMode === 'split' ? 'bg-blue-600 text-white' : 'bg-slate-800/50 text-slate-400 hover:bg-slate-800'}`}
             >
               {viewMode === 'chat' ? <SplitSquareVertical size={18} /> : <MessageSquare size={18} />}
             </button>
           )}

           {viewMode === 'archive' ? (
                <button onClick={() => setViewMode('chat')} className="p-2 rounded-full bg-slate-800 text-slate-300 hover:bg-slate-700">
                    <X size={20} />
                </button>
           ) : (
             <>
               <div className={`px-2.5 py-1 rounded-full text-[10px] font-bold border ${isListenModeActive ? 'bg-orange-500/10 border-orange-500/20 text-orange-400' : isOfflineActive ? 'bg-red-500/10 border-red-500/20 text-red-400' : 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'}`}>
                {isOfflineActive ? 'OFFLINE' : 'ONLINE'}
               </div>
               
               <button onClick={() => setShowSettings(true)} className={`p-2 rounded-full transition-colors active:scale-95 ${isListenModeActive ? 'bg-orange-800/50 hover:bg-orange-800 text-orange-300' : 'bg-slate-800/50 hover:bg-slate-800 text-slate-400'}`}>
                <Settings size={20} />
              </button>
             </>
           )}
        </div>
      </header>
      
      {/* Direction Indicator (Only in Chat Mode) */}
      {viewMode === 'chat' && !isListenModeActive && (
      <div className="absolute top-16 left-0 right-0 z-10 flex justify-center mt-2 animate-fade-in-down pointer-events-auto">
          <button 
             onClick={toggleOfflineDirection}
             disabled={!isOfflineActive} 
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

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col z-10 relative overflow-hidden">
        
        {/* --- VIEW MODE: LISTEN (TRANSCRIPTION FEED) --- */}
        {viewMode === 'listen' ? (
            <div 
               ref={chatContainerRef} 
               onScroll={handleScroll}
               className="flex-1 overflow-y-auto px-6 pt-10 pb-20 scroll-smooth no-scrollbar"
            >
                {/* Realtime Input (Original Language Preview) */}
                {realtimeInput && (
                    <div className="mb-8 opacity-50 transition-opacity">
                        <span className="text-xs text-orange-300/70 font-bold uppercase tracking-widest mb-2 block flex items-center gap-2">
                             <Ear size={12} className="animate-pulse"/> Algılanıyor ({targetDetails.short})
                        </span>
                        <p className="text-2xl text-orange-100/50 font-light leading-relaxed italic">
                            {realtimeInput}...
                        </p>
                    </div>
                )}

                {/* Transcriptions */}
                {messages.filter(m => m.role === 'model').map((msg) => (
                    <div key={msg.id} className="mb-8 animate-fade-in-up">
                        <span className="text-xs text-orange-400 font-bold uppercase tracking-widest mb-1 block flex items-center gap-2">
                             {sourceDetails.flag} Çeviri
                        </span>
                        <p className="text-2xl md:text-3xl font-medium text-white leading-relaxed drop-shadow-sm">
                            {msg.text}
                        </p>
                    </div>
                ))}
                
                 {/* Realtime Output (Draft Translation) */}
                 {realtimeOutput && (
                    <div className="mb-4">
                        <p className="text-2xl md:text-3xl font-medium text-orange-200 leading-relaxed blur-[1px]">
                            {realtimeOutput}
                        </p>
                    </div>
                 )}

                <div className="h-32"></div> {/* Spacer */}
            </div>
        ) : viewMode === 'archive' ? (
            /* --- VIEW MODE: ARCHIVE --- */
            <div className="flex-1 overflow-y-auto px-4 pt-4 pb-20">
                {openedArchive ? (
                     <div className="space-y-4 animate-fade-in">
                        <button onClick={() => setOpenedArchive(null)} className="flex items-center gap-2 text-slate-400 text-sm mb-4">
                            <ChevronRight className="rotate-180" size={16} /> Geri Dön
                        </button>
                        <div className="bg-slate-900 p-4 rounded-xl border border-slate-800 mb-4">
                            <div className="flex justify-between items-start">
                                <div>
                                    <h3 className="text-white font-bold text-lg flex items-center gap-2">
                                        <Calendar size={16} className="text-blue-400" />
                                        {new Date(openedArchive.date).toLocaleString('tr-TR')}
                                    </h3>
                                    <span className="text-slate-500 text-xs mt-1 block">
                                        Hedef Dil: {getLangDetails(openedArchive.targetLang).name} {getLangDetails(openedArchive.targetLang).flag}
                                    </span>
                                </div>
                                <button onClick={() => deleteArchivedSession(openedArchive.id)} className="p-2 bg-red-500/10 text-red-400 rounded-lg">
                                    <Trash2 size={16} />
                                </button>
                            </div>
                        </div>
                        <div className="space-y-6">
                            {openedArchive.messages.map((msg, i) => (
                                <div key={i} className="bg-slate-800/50 p-4 rounded-xl border border-slate-700/50">
                                    <p className="text-lg text-slate-200 leading-relaxed">{msg.text}</p>
                                </div>
                            ))}
                        </div>
                     </div>
                ) : (
                    <div className="space-y-3">
                        {savedSessions.length === 0 ? (
                            <div className="flex flex-col items-center justify-center h-64 text-slate-500 opacity-50">
                                <FolderOpen size={48} />
                                <p className="mt-4 text-sm">Henüz kaydedilmiş bir oturum yok.</p>
                            </div>
                        ) : (
                            savedSessions.map(session => (
                                <button 
                                   key={session.id} 
                                   onClick={() => setOpenedArchive(session)}
                                   className="w-full bg-slate-800 p-4 rounded-xl border border-slate-700 flex items-center justify-between hover:bg-slate-750 transition-colors group"
                                >
                                    <div className="flex items-start gap-3 text-left">
                                        <div className="p-3 bg-blue-600/20 text-blue-400 rounded-lg">
                                            <FileText size={20} />
                                        </div>
                                        <div>
                                            <div className="font-semibold text-slate-200 text-sm">
                                                {new Date(session.date).toLocaleDateString('tr-TR', { day: 'numeric', month: 'long', hour: '2-digit', minute:'2-digit' })}
                                            </div>
                                            <div className="text-xs text-slate-500 mt-1 line-clamp-1">
                                                {session.preview}
                                            </div>
                                            <div className="mt-1.5 flex gap-2">
                                                <span className="text-[10px] bg-slate-700 px-1.5 py-0.5 rounded text-slate-300">
                                                    {getLangDetails(session.targetLang).short}
                                                </span>
                                            </div>
                                        </div>
                                    </div>
                                    <ChevronRight className="text-slate-600 group-hover:text-white transition-colors" size={20} />
                                </button>
                            ))
                        )}
                    </div>
                )}
            </div>
        ) : viewMode === 'split' ? (
             /* --- VIEW MODE: SPLIT --- */
             <div className="flex-1 flex flex-col items-stretch h-full pb-[calc(8rem+env(safe-area-inset-bottom))]">
                <div className="flex-1 bg-slate-900/50 border-b border-slate-800 p-6 flex flex-col justify-center items-center relative rotate-180">
                     <div className="absolute top-4 left-4 flex items-center gap-2 opacity-50">
                        <span className="text-2xl">{targetDetails.flag}</span>
                        <span className="text-sm font-bold">{targetDetails.name}</span>
                     </div>
                     <div className="text-center">
                         <div className="text-3xl font-bold text-emerald-300 leading-snug">
                            {isInputTarget ? realtimeInput : (realtimeOutput || lastModelMsg?.text || "...")}
                         </div>
                     </div>
                </div>
                <div className="flex-1 bg-slate-950/50 p-6 flex flex-col justify-center items-center relative">
                     <div className="absolute top-4 left-4 flex items-center gap-2 opacity-50">
                        <span className="text-2xl">{sourceDetails.flag}</span>
                        <span className="text-sm font-bold">{sourceDetails.name}</span>
                     </div>
                     <div className="text-center">
                         <div className="text-3xl font-bold text-slate-100 leading-snug">
                            {!isInputTarget ? realtimeInput : (realtimeOutput || lastUserMsg?.text || "...")}
                         </div>
                     </div>
                </div>
             </div>
        ) : (
        /* --- VIEW MODE: CHAT (STANDARD) --- */
        <div ref={chatContainerRef} onScroll={handleScroll} className="flex-1 overflow-y-auto px-4 pt-10 pb-6 space-y-6 scroll-smooth no-scrollbar">
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
                   {isNoiseMode ? 'Bas-konuş modu aktif. Konuşmak için butona basılı tutun.' : 'Konuşmaya başlamak için mikrofonu kullanın.'}
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
                      <> <Mic size={10} /> <span>Konuşmacı</span> </>
                  ) : (
                      <> 
                        {/* Use Ear icon if it was a listen mode message (implied by context usually, simplified here) */}
                        <Sparkles size={10} /> <span>Çevirmen</span> 
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

          {/* Real-time Bubbles (Chat Mode) */}
          {(realtimeInput || realtimeOutput) && (
             <div className="space-y-4">
                {realtimeInput && (
                  <div className="flex flex-col items-start opacity-70">
                     <span className="text-[10px] text-blue-400 mb-1 px-1 animate-pulse flex items-center gap-1"><Mic size={10}/> 
                     {/* Heuristic display */}
                     {getLangDetails(detectedRealtimeInputLang).short}...
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
        <div className={`backdrop-blur-xl border-t px-6 pt-4 z-30 pb-[calc(1.5rem+env(safe-area-inset-bottom,20px))] absolute bottom-0 w-full transition-colors duration-500 ${isListenModeActive ? 'bg-orange-950/80 border-orange-900/50' : 'bg-slate-950/80 border-slate-800/50'}`}>
          
          {/* Visualizer Bar (Above Dock) */}
          <div className="h-10 w-full flex items-center justify-center mb-4">
            {isConnected ? (
              <div className="w-full h-full opacity-80">
                 <AudioVisualizer analyser={inputAnalyserRef.current} isActive={true} color={isListenModeActive ? '#fb923c' : isOfflineActive ? '#f97316' : '#4ade80'} />
              </div>
            ) : (
              <div className="h-1 w-16 bg-slate-800 rounded-full"></div>
            )}
          </div>

          <div className="flex items-center justify-between max-w-sm mx-auto">
            
            {/* Left: Source Language Selector */}
            <div className="w-20 flex flex-col items-center gap-1">
               <button 
                 onClick={() => setSelectorType('source')}
                 // Disable source selection in Listen Mode (Always Turkish output)
                 disabled={isListenModeActive} 
                 className={`w-full h-14 border rounded-2xl flex items-center justify-center transition-all active:scale-95 touch-manipulation relative overflow-hidden group ${isListenModeActive ? 'bg-orange-900/40 border-orange-800/50 opacity-80' : 'bg-slate-800 border-slate-700 hover:border-slate-500'}`}
               >
                  <span className="text-3xl filter drop-shadow-md z-10 transition-transform group-active:scale-90">{sourceDetails.flag}</span>
               </button>
               <span className={`text-[9px] font-medium ${isListenModeActive ? 'text-orange-400' : 'text-slate-500'}`}>Kaynak</span>
            </div>

            {/* Center Controls (Two Buttons) */}
            <div className="relative -mt-6 flex gap-4 items-end">
                
                {/* 1. LISTEN BUTTON (DINLE) */}
                <div className="relative group">
                    {/* Orange Halo for Listen Mode */}
                    {isListenModeActive && (
                        <>
                          <div className="absolute inset-0 rounded-full animate-[ping_2s_cubic-bezier(0,0,0.2,1)_infinite] bg-orange-500/50"></div>
                          <div className="absolute inset-0 rounded-full animate-[ping_2s_cubic-bezier(0,0,0.2,1)_infinite_200ms] bg-orange-500/30"></div>
                        </>
                    )}
                    <button
                        // Noise Mode Handling for Listen Button
                        onMouseDown={isNoiseMode ? () => handlePTTStart('listen') : undefined}
                        onMouseUp={isNoiseMode ? handlePTTEnd : undefined}
                        onTouchStart={isNoiseMode ? (e) => { e.preventDefault(); handlePTTStart('listen'); } : undefined}
                        onTouchEnd={isNoiseMode ? (e) => { e.preventDefault(); handlePTTEnd(); } : undefined}
                        onClick={!isNoiseMode ? handleListenToggle : undefined}

                        className={`relative w-14 h-14 rounded-full flex items-center justify-center shadow-lg transition-all duration-300 transform active:scale-95 z-10 border-2 select-none ${
                            isListenModeActive 
                            ? isNoiseMode && isHoldingMic 
                                ? 'bg-orange-500 text-white border-orange-300 scale-110 shadow-[0_0_20px_rgba(251,146,60,0.6)]' // Holding visual
                                : 'bg-orange-600 text-white border-orange-400 scale-110' 
                            : 'bg-slate-800 text-slate-400 border-slate-700 hover:bg-slate-700'
                        }`}
                    >
                        {isListenModeActive && isConnecting ? <Loader2 size={24} className="animate-spin" /> : <Ear size={24} />}
                    </button>
                    <span className="absolute -bottom-5 left-1/2 -translate-x-1/2 text-[9px] font-bold text-slate-500 uppercase tracking-wide">
                        {isNoiseMode ? 'Bas-Tut' : 'Dinle'}
                    </span>
                </div>

                {/* 2. MAIN CHAT BUTTON */}
                <div className="relative group">
                    {/* Green/Red Halo for Chat Mode */}
                    {isConnected && !isListenModeActive && (
                        <>
                          <div className={`absolute inset-0 rounded-full animate-[ping_2s_cubic-bezier(0,0,0.2,1)_infinite] ${isOfflineActive ? 'bg-red-500/40' : 'bg-emerald-500/40'}`}></div>
                          <div className={`absolute inset-0 rounded-full animate-[ping_2s_cubic-bezier(0,0,0.2,1)_infinite_200ms] ${isOfflineActive ? 'bg-red-500/30' : 'bg-emerald-500/30'}`}></div>
                        </>
                    )}
                    <button
                        // Noise Mode Handling for Chat Button
                        onMouseDown={isNoiseMode ? () => handlePTTStart('bidirectional') : undefined}
                        onMouseUp={isNoiseMode ? handlePTTEnd : undefined}
                        onTouchStart={isNoiseMode ? (e) => { e.preventDefault(); handlePTTStart('bidirectional'); } : undefined}
                        onTouchEnd={isNoiseMode ? (e) => { e.preventDefault(); handlePTTEnd(); } : undefined}
                        onClick={!isNoiseMode ? handleMicToggle : undefined}
                        
                        className={`relative w-16 h-16 rounded-full flex items-center justify-center shadow-2xl transition-all duration-300 transform active:scale-95 z-10 select-none ${
                          isConnecting && !isListenModeActive
                            ? 'bg-slate-800 border-4 border-slate-700 cursor-wait'
                            : isConnected && !isListenModeActive
                              ? isOfflineActive 
                                ? 'bg-red-600 text-white shadow-red-600/40 ring-4 ring-red-900/50 scale-105' 
                                : isNoiseMode && isHoldingMic
                                    ? 'bg-emerald-500 text-white shadow-[0_0_30px_rgba(16,185,129,0.8)] ring-4 ring-emerald-300 scale-110' // Holding visual
                                    : 'bg-emerald-600 text-white shadow-emerald-600/40 ring-4 ring-emerald-900/50 scale-105'
                              : 'bg-slate-100 text-slate-900 hover:bg-white border-4 border-slate-300 shadow-white/10'
                        }`}
                    >
                        {isConnecting && !isListenModeActive ? (
                           <Loader2 size={28} className="animate-spin text-slate-400" />
                        ) : isConnected && !isListenModeActive ? (
                           // Icon change based on hold state in noise mode
                           isNoiseMode && isHoldingMic ? <Waves size={28} className="animate-pulse" /> : <Square size={24} fill="currentColor" className="rounded-sm" />
                        ) : (
                           // Icon change based on mode
                           isNoiseMode ? <Mic size={28} strokeWidth={2.5} className="text-slate-900" /> : <Mic size={28} strokeWidth={2.5} />
                        )}
                    </button>
                    <span className="absolute -bottom-5 left-1/2 -translate-x-1/2 text-[9px] font-bold text-slate-500 uppercase tracking-wide">
                        {isNoiseMode ? 'Bas-Konuş' : 'Sohbet'}
                    </span>
                </div>

            </div>

            {/* Right: Target Language Selector */}
            <div className="w-20 flex flex-col items-center gap-1">
               <button 
                 onClick={() => setSelectorType('target')}
                 className={`w-full h-14 border rounded-2xl flex items-center justify-center transition-all active:scale-95 touch-manipulation relative overflow-hidden group ${isListenModeActive ? 'bg-orange-900/40 border-orange-800/50 text-orange-200' : 'bg-slate-800 border-slate-700 text-slate-200 hover:border-slate-500'}`}
               >
                  <span className="text-3xl filter drop-shadow-md z-10 transition-transform group-active:scale-90">{targetDetails.flag}</span>
               </button>
               <span className={`text-[9px] font-medium ${isListenModeActive ? 'text-orange-400' : 'text-slate-500'}`}>Hedef</span>
            </div>

          </div>
        </div>

      </main>

      {/* SAVE SESSION MODAL */}
      {showSaveModal && (
          <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-6 animate-fade-in">
              <div className="bg-slate-900 border border-slate-700 w-full max-w-sm rounded-2xl shadow-2xl p-6">
                  <div className="w-12 h-12 bg-blue-500/20 rounded-full flex items-center justify-center mb-4 text-blue-400 mx-auto">
                      <Save size={24} />
                  </div>
                  <h3 className="text-xl font-bold text-white text-center mb-2">Oturumu Kaydet?</h3>
                  <p className="text-slate-400 text-center text-sm mb-6">
                      Dinleme modunu kapatıyorsunuz. Bu oturumdaki çevirileri daha sonra incelemek için arşive kaydetmek ister misiniz?
                  </p>
                  <div className="flex gap-3">
                      <button 
                        onClick={discardSession}
                        className="flex-1 py-3 rounded-xl bg-slate-800 text-slate-300 font-medium hover:bg-slate-700"
                      >
                          Sil
                      </button>
                      <button 
                        onClick={saveSession}
                        className="flex-1 py-3 rounded-xl bg-blue-600 text-white font-bold hover:bg-blue-500 shadow-lg shadow-blue-900/20"
                      >
                          Kaydet
                      </button>
                  </div>
              </div>
          </div>
      )}

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
                      ? isListenModeActive ? 'text-orange-400' : 'text-emerald-400' 
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
               
               {/* Archive Button */}
               <button 
                  onClick={() => { setViewMode('archive'); setShowSettings(false); }}
                  className="w-full bg-slate-800 p-4 rounded-xl border border-slate-700 flex items-center justify-between hover:bg-slate-750 transition-colors"
               >
                   <div className="flex items-center gap-3">
                       <FolderOpen size={20} className="text-orange-400" />
                       <div className="text-left">
                           <div className="font-semibold text-white">Kayıtlı Oturumlar</div>
                           <div className="text-xs text-slate-500">Arşivlenmiş dinleme kayıtlarını incele</div>
                       </div>
                   </div>
                   <ChevronRight size={18} className="text-slate-500" />
               </button>

               {/* API Key Management */}
               <div className="bg-slate-800/30 p-4 rounded-2xl border border-slate-800">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-semibold text-slate-200 flex items-center gap-2">
                        <Key size={18} className="text-yellow-400" /> API Anahtarı
                    </span>
                    <button 
                        onClick={handleDeleteApiKey} 
                        className="px-3 py-1 bg-red-500/10 text-red-400 text-xs font-bold rounded-lg hover:bg-red-500/20 flex items-center gap-1"
                    >
                        <LogOut size={12} /> Çıkış
                    </button>
                  </div>
                  <p className="text-xs text-slate-500 mb-2">
                      Mevcut anahtar: <span className="text-slate-300 font-mono">••••••••{apiKey.slice(-4)}</span>
                  </p>
                  <p className="text-[10px] text-slate-600">
                      Bu anahtar sadece cihazınızda saklanır ve Google sunucularına iletilir.
                  </p>
               </div>

               {/* Noise Mode Toggle */}
               <div className="bg-slate-800/30 p-4 rounded-2xl border border-slate-800">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-semibold text-slate-200 flex items-center gap-2">
                        <Waves size={18} className="text-teal-400" /> Gürültü Engelleme (Bas-Konuş)
                    </span>
                    <button onClick={toggleNoiseMode} 
                      className={`w-12 h-7 rounded-full p-1 transition-colors ${isNoiseMode ? 'bg-teal-500' : 'bg-slate-700'}`}>
                      <div className={`w-5 h-5 rounded-full bg-white transition-transform ${isNoiseMode ? 'translate-x-5' : ''}`}></div>
                    </button>
                  </div>
                  <p className="text-xs text-slate-500">
                      Kalabalık ortamlarda sadece butona bastığınızda sesi iletir. Arka plan gürültüsünü filtreler.
                  </p>
               </div>

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
                       let flag = '🏳️';
                       if (pack.id.includes('en')) flag = '🇬🇧';
                       if (pack.id.includes('nl')) flag = '🇳🇱';
                       if (pack.id.includes('ar')) flag = '🇸🇦';
                       if (pack.id.includes('ru')) flag = '🇷🇺';
                       if (pack.id.includes('de')) flag = '🇩🇪';
                       if (pack.id.includes('it')) flag = '🇮🇹';
                       if (pack.id.includes('fr')) flag = '🇫🇷';
                       if (pack.id.includes('ua')) flag = '🇺🇦';
                       
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