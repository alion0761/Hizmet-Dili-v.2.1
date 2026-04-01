import React, { useState, useRef, useCallback, useEffect } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality, FunctionDeclaration, Type } from '@google/genai';
import { X, Loader2, BookOpen, Mic, Trash2, MessageSquare } from 'lucide-react';
import { float32To16BitPCM, arrayBufferToBase64, pcm16ToFloat32 } from '../utils/audioUtils';
import { db, auth } from '../firebase';
import { collection, onSnapshot, addDoc, query, orderBy, serverTimestamp } from 'firebase/firestore';
import { onAuthStateChanged } from 'firebase/auth';

interface LearnedWord {
  id: string;
  original: string;
  translation: string;
}

interface EducationCoachProps {
  onClose: () => void;
  apiKey: string;
}

const EducationCoach: React.FC<EducationCoachProps> = ({ onClose, apiKey }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [learnedWords, setLearnedWords] = useState<LearnedWord[]>([]);
  const [userId, setUserId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'coach' | 'learned'>('coach');
  
  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const inputAnalyserRef = useRef<AnalyserNode | null>(null);
  const outputAnalyserRef = useRef<AnalyserNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const sessionPromiseRef = useRef<Promise<any> | null>(null);
  const activeSessionRef = useRef<any>(null);
  const nextStartTimeRef = useRef<number>(0);
  const currentAudioSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const lastInterruptionTimeRef = useRef<number>(0);

  useEffect(() => {
    const unsubscribeAuth = onAuthStateChanged(auth, (user) => {
      setUserId(user?.uid || null);
    });
    return () => unsubscribeAuth();
  }, []);

  useEffect(() => {
    if (!userId) {
      setLearnedWords([]);
      return;
    }

    const q = query(collection(db, 'users', userId, 'learnedWords'), orderBy('createdAt', 'desc'));
    const unsubscribe = onSnapshot(q, (snapshot) => {
      const words: LearnedWord[] = [];
      snapshot.forEach((doc) => {
        words.push({ id: doc.id, ...doc.data() } as LearnedWord);
      });
      setLearnedWords(words);
    });

    return () => unsubscribe();
  }, [userId]);

  const addLearnedWord = async (original: string, translation: string) => {
    if (!userId) return;
    try {
      await addDoc(collection(db, 'users', userId, 'learnedWords'), {
        original,
        translation,
        userId,
        createdAt: serverTimestamp()
      });
    } catch (e) {
      console.error('Error adding word:', e);
    }
  };

  const addLearnedWordDeclaration: FunctionDeclaration = {
    name: "addLearnedWord",
    description: "Yeni öğrenilen bir kelimeyi veya cümleyi listeye ekle.",
    parameters: {
      type: Type.OBJECT,
      properties: {
        original: { type: Type.STRING, description: "Felemenkçe kelime veya cümle." },
        translation: { type: Type.STRING, description: "Türkçe karşılığı." }
      },
      required: ["original", "translation"]
    }
  };

  const stopPlayback = useCallback(() => {
    if (currentAudioSourceRef.current) {
      try {
        currentAudioSourceRef.current.stop();
      } catch (e) {
        // Source might already be stopped
      }
      currentAudioSourceRef.current = null;
    }
    setIsSpeaking(false);
    // Reset nextStartTime to current time to avoid gaps after interruption
    if (outputAudioContextRef.current) {
      nextStartTimeRef.current = outputAudioContextRef.current.currentTime;
    }
  }, []);

  const stopConnection = useCallback(() => {
    setIsConnecting(false); setIsConnected(false); setIsSpeaking(false);
    stopPlayback();
    if (mediaStreamRef.current) mediaStreamRef.current.getTracks().forEach(t => t.stop());
    if (scriptProcessorRef.current) scriptProcessorRef.current.disconnect();
    if (inputAudioContextRef.current && inputAudioContextRef.current.state !== 'closed') inputAudioContextRef.current.close();
    if (outputAudioContextRef.current && outputAudioContextRef.current.state !== 'closed') outputAudioContextRef.current.close();
    if (activeSessionRef.current) activeSessionRef.current.close();
    sessionPromiseRef.current = null;
  }, []);

  const startCoach = async () => {
    if (isConnected) {
      stopConnection();
      return;
    }
    setIsConnecting(true);

    try {
      const aiClient = new GoogleGenAI({ apiKey });
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      
      const audioCtx = new AudioContext({ sampleRate: 16000 });
      await audioCtx.resume();
      inputAudioContextRef.current = audioCtx;
      
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      inputAnalyserRef.current = analyser;
      const scriptProcessor = audioCtx.createScriptProcessor(4096, 1, 1);
      scriptProcessorRef.current = scriptProcessor;
      
      scriptProcessor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        
        // Voice Activity Detection (VAD) for interruption
        let sum = 0;
        for (let i = 0; i < inputData.length; i++) {
          sum += inputData[i] * inputData[i];
        }
        const rms = Math.sqrt(sum / inputData.length);
        
        // If user speaks while model is speaking, stop playback
        if (rms > 0.05 && isSpeaking) {
          const now = Date.now();
          // Debounce interruption to avoid accidental triggers
          if (now - lastInterruptionTimeRef.current > 500) {
            stopPlayback();
            lastInterruptionTimeRef.current = now;
            // Optionally send a signal to the model that it was interrupted
            // The Live API handles this naturally if we stop sending audio or send a specific message
            // but simply stopping local playback is usually enough for the user experience.
          }
        }

        const pcmData = float32To16BitPCM(inputData);
        sessionPromiseRef.current?.then(s => s.sendRealtimeInput({ 
          audio: { mimeType: 'audio/pcm;rate=16000', data: arrayBufferToBase64(pcmData) } 
        }));
      };

      source.connect(analyser);
      analyser.connect(scriptProcessor);
      scriptProcessor.connect(audioCtx.destination);

      const outCtx = new AudioContext({ sampleRate: 24000 });
      await outCtx.resume();
      outputAudioContextRef.current = outCtx;
      const outAnalyser = outCtx.createAnalyser();
      outputAnalyserRef.current = outAnalyser;
      outAnalyser.connect(outCtx.destination);

      const sessionPromise = aiClient.live.connect({
        model: 'gemini-3.1-flash-live-preview',
        config: {
          systemInstruction: `Sen, kullanıcının Felemenkçe öğrenme koçusun.
Rolün yalnızca sesli iletişim kurmaktır. KESİNLİKLE YAZILI METİN ÜRETME, YALNIZCA SESLİ KONUŞ.
Yeni bir kelime veya cümle öğrettiğinde, 'addLearnedWord' aracını kullanarak bunu listeye ekle.

ÖNEMLİ: Eğer kullanıcı senin sözünü keserse (sen konuşurken araya girerse), buna komik, şakacı ve hafif sitemkar bir şekilde tepki ver. 
Örneğin: "Hey! Daha cümlem bitmemişti!", "Sözümü balla değil, Felemenkçe ile kestin bakıyorum!", "Tam da en heyecanlı yerindeydim, neden böldün ki şimdi?" gibi esprili sitemlerde bulun. 
Sitemden sonra hemen konuya dön ve kullanıcının ne dediğine cevap ver.

Rolün: arkadaş canlısı, eğlenceli, neşeli, motive edici ve şakacı bir şekilde kullanıcıya sürekli destek olmaktır. Kullanıcı seninle rahatça konuşabilmeli, soru sorabilmeli ve öğrenme sürecinde kendini yalnız hissetmemelidir.

Felemenkçe telaffuzuna çok dikkat et. Gerçek bir Hollandalı gibi konuş. Vurgulara, tonlamalara ve sesletime (artikülasyon) özen göster.

Temel Kimliğin
Kullanıcıya karşı her zaman samimi, sıcak, destekleyici ve arkadaş gibi davran.
Üslubun eğlenceli, hafif şakacı ve motive edici olsun.
Asla sıkıcı, aşırı resmi veya soğuk bir öğretmen gibi davranma.
Kullanıcı hata yaptığında onu eleştirme; nazikçe düzelt ve teşvik et.
Öğrenme sürecini keyifli hale getirmeye çalış.
Ana Görevin
Kullanıcının Felemenkçe öğrenmesine yardım et. Bunu yaparken:

Kelime anlamları ver,
Cümle çevirileri yap,
Telaffuz desteği sun,
Basit açıklamalar yap,
Gerektiğinde örnek cümleler üret,
Kullanıcıya mini alıştırmalar ve hatırlatma soruları sor.
Telaffuz Kuralı
Kullanıcı bir kelime veya cümle anlamı sorduğunda:

Önce Felemenkçe karşılığını ver,
Hemen ardından Türkçe okunuşa yakın telaffuzunu sesli söyle,
Sonra kısa ve anlaşılır bir açıklama ekle,
Mümkünse 1 kısa örnek cümle ver.
Telaffuzları, kullanıcının kolay anlayabileceği şekilde sesli olarak ver.
Gerekirse çok kısa bir not ekle: örneğin “gırtlaktan söylenir”, “hafif yuvarlanır”, “vurgu ilk hecede” gibi.

Hafıza ve Takip Davranışı
Konuşma boyunca kullanıcıyla daha önce çalıştığın kelimeleri, cümleleri, hataları, ilgi alanlarını ve zorluk yaşadığı noktaları hatırlıyor gibi davran ve bunları sonraki cevaplarında kullan.

Daha önce öğrenilen kelimeleri uygun anlarda tekrar gündeme getir.
Ara sıra kullanıcıya küçük hatırlatma soruları sor:
“Geçen öğrendiğimiz ‘goedemorgen’ ne demekti?”
“Hatırlıyor musun, ‘dank je wel’ nasıl okunuyordu?”
Kullanıcının seviyesine göre tekrar yap.
Ancak bunu bunaltıcı şekilde değil, doğal sohbet akışı içinde yap.
Öğretim Tarzı
Açıklamaları kısa, net ve anlaşılır yap.
Gereksiz dilbilgisi yüklemesi yapma; kullanıcı isterse detay ver.
Önceliğin pratik kullanım olsun.
Kullanıcının seviyesini anlamaya çalış ve ona göre konuş.
Başlangıç seviyesinde ise basit kelimeler ve günlük kalıplarla ilerle.
Gerektiğinde küçük quizler yap.
Uzun anlatım yerine mini mini öğretim blokları kullan.
Sohbet Tarzı
Cevapların doğal, arkadaş canlısı ve akıcı olsun.
Yer yer hafif espriler yapabilirsin ama öğretici tarafı gölgelememelisin.
Kullanıcı motivasyon kaybı yaşarsa onu destekle:
“Harika gidiyorsun!”
“Bak, bu kelime artık sende oturmaya başladı 😄”
“Yanlış yapman çok normal, beynin şu an Felemenkçe kas yapıyor.”
Soru Sorma Davranışı
Uygun anlarda kullanıcıya kısa sorular sor:

“Bunu bir de sen cümle içinde kullanmak ister misin?”
“Mini test gelsin mi?”
“Bu kelimeyi yarın da hatırlaman için sana küçük bir ipucu vereyim mi?”
“Geçen çalıştığımız kelimeyi hatırlıyor musun?”
Ama her cevapta soru sorma; doğal bir denge kur.

Yanıt Biçimi
Cevaplarını kullanıcı dostu biçimde düzenle:

Kısa paragraflar kullan,
Öğretici ama samimi bir ton kullan.
Kullanıcının Özel İstekleri
Eğer kullanıcı:

“Sadece çeviri ver” derse kısa cevap ver.
“Detaylı anlat” derse daha fazla açıklama yap.
“Beni test et” derse quiz moduna geç.
“Sadece Felemenkçe konuş” derse seviyesine uygun şekilde büyük ölçüde Felemenkçe kullan.
“Telaffuzu tekrar söyle” derse daha açık telaffuz ver.
Hata Düzeltme Kuralı
Kullanıcı yanlış yaparsa veya yanlış çeviri yaparsa:

Önce nazikçe doğru halini ver,
Sonra çok kısa nedenini açıkla,
Sonra motive edici bir cümle ekle.
Örnek yaklaşım:
“Minik bir düzeltme yapayım 😄”
“Doğrusu şöyle: …”
“Ama mantığı doğru kurmuşsun, bu çok iyi.”
Örnek Cevap Tarzı
Kullanıcı: “Günaydın ne demek?”
Sen:
“Goedemorgen. Okunuşu: hudımorgın. Günaydın demek. Sabah selamlaşmalarında kullanılır. Örnek: Goedemorgen, hoe gaat het?”

Hatırlatma Davranışı Örneği
Uygun bir zamanda şöyle diyebilirsin:

“Bu arada mini hatırlatma 😄 ‘Goedemorgen’ ne demekti, hatırlıyor musun?”
“Geçen çalıştığımız ‘alsjeblieft’ kelimesini tekrar edelim mi?”
Sınırlar
Yanlış veya uydurma bilgi verme.
Emin olmadığın telaffuzlarda bunu belirt ve en yakın okunuşu ver.
Kullanıcıyı yargılama, küçümseme veya sıkma.
Ana odağın her zaman Felemenkçe öğrenimini desteklemek olsun.
Genel Amaç
Kullanıcının seni:

bir öğretmen,
bir çalışma arkadaşı,
bir motivasyon koçu,
ve gerektiğinde eğlenceli bir dil partneri
olarak hissetmesini sağla.
Her zaman hedefin şu olsun:
Kullanıcı Felemenkçe öğrenirken hem ilerlediğini hissetsin hem de keyif aldığını sesli olarak duysun.`,
          responseModalities: [Modality.AUDIO],
          tools: [{ functionDeclarations: [addLearnedWordDeclaration] }]
        },
        callbacks: {
          onopen: () => { setIsConnecting(false); setIsConnected(true); },
          onmessage: (msg: LiveServerMessage) => {
            if (msg.toolCall) {
              for (const call of msg.toolCall.functionCalls) {
                if (call.name === 'addLearnedWord') {
                  addLearnedWord(call.args.original as string, call.args.translation as string);
                  activeSessionRef.current?.sendToolResponse({
                    functionResponses: [{
                      name: 'addLearnedWord',
                      id: call.id,
                      response: { result: 'success' }
                    }]
                  });
                }
              }
            }
            const audio = msg.serverContent?.modelTurn?.parts?.find(p => p.inlineData)?.inlineData?.data;
            if (audio) {
              setIsSpeaking(true);
              playAudio(audio);
            }
          },
          onclose: stopConnection,
          onerror: (e) => { console.error('Live API error:', e); stopConnection(); }
        }
      });
      sessionPromiseRef.current = sessionPromise;
      sessionPromise.then(s => activeSessionRef.current = s);
    } catch (e: any) { console.error('Error starting coach:', e); stopConnection(); }
  };

  const clearHistory = async () => {
    stopConnection();
    await startCoach();
  };

  const playAudio = async (base64: string) => {
    const ctx = outputAudioContextRef.current;
    if (!ctx) return;
    const arrayBuffer = Uint8Array.from(atob(base64), c => c.charCodeAt(0)).buffer;
    const float32Data = pcm16ToFloat32(arrayBuffer);
    const buffer = ctx.createBuffer(1, float32Data.length, 24000);
    buffer.getChannelData(0).set(float32Data);
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(outputAnalyserRef.current!);
    source.onended = () => {
      if (currentAudioSourceRef.current === source) {
        setIsSpeaking(false);
        currentAudioSourceRef.current = null;
      }
    };
    currentAudioSourceRef.current = source;
    source.start(Math.max(nextStartTimeRef.current, ctx.currentTime));
    nextStartTimeRef.current = Math.max(nextStartTimeRef.current, ctx.currentTime) + buffer.duration;
  };

  return (
    <div className="fixed inset-0 bg-gray-50 z-50 flex flex-col">
      {/* Header */}
      <header className="flex items-center justify-between p-4 bg-white border-b border-gray-200">
        <h1 className="text-xl font-bold text-gray-800 flex items-center gap-2">
          <MessageSquare className="text-orange-600" /> Eğitim Koçu
        </h1>
        <button onClick={onClose} className="p-2 text-gray-500 hover:text-gray-800 transition-colors">
          <X size={24} />
        </button>
      </header>

      {/* Tabs (Mobile Only) */}
      <div className="flex md:hidden border-b border-gray-200">
        <button 
          onClick={() => setActiveTab('coach')}
          className={`flex-1 py-3 text-sm font-medium ${activeTab === 'coach' ? 'text-orange-600 border-b-2 border-orange-600' : 'text-gray-500'}`}
        >
          Koç
        </button>
        <button 
          onClick={() => setActiveTab('learned')}
          className={`flex-1 py-3 text-sm font-medium ${activeTab === 'learned' ? 'text-orange-600 border-b-2 border-orange-600' : 'text-gray-500'}`}
        >
          Öğrendiklerim ({learnedWords.length})
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
        {/* Coach Section */}
        <div className={`flex-1 flex flex-col items-center justify-center p-6 ${activeTab === 'coach' ? 'flex' : 'hidden md:flex'}`}>
          <div className="relative w-64 h-64 flex items-center justify-center">
            {isSpeaking && (
              <>
                <div className="absolute w-64 h-64 bg-orange-200 rounded-full animate-ping opacity-50"></div>
                <div className="absolute w-48 h-48 bg-orange-300 rounded-full animate-ping opacity-40 delay-100"></div>
              </>
            )}
            <div className="w-40 h-40 rounded-full bg-gradient-to-tr from-blue-500 via-purple-500 to-orange-500 animate-spin p-1 z-10">
              <div className="w-full h-full rounded-full bg-white flex items-center justify-center">
                <Mic size={48} className="text-gray-700" />
              </div>
            </div>
          </div>
          
          <div className="flex flex-col items-center gap-4 mt-12 w-full max-w-xs">
            <button
              onClick={startCoach}
              className={`w-full py-4 rounded-xl text-white font-bold text-lg shadow-lg transition-all flex items-center justify-center gap-2 ${isConnected ? 'bg-red-600 hover:bg-red-700' : 'bg-orange-600 hover:bg-orange-700'}`}
            >
              {isConnecting ? <Loader2 size={24} className="animate-spin" /> : isConnected ? 'Koçu Durdur' : 'Eğitimi Başlat'}
            </button>
            <button
              onClick={clearHistory}
              className={`w-full py-3 rounded-xl text-gray-600 font-medium bg-gray-100 hover:bg-gray-200 transition-all flex items-center justify-center gap-2 ${isConnected ? 'opacity-100' : 'opacity-50 pointer-events-none'}`}
            >
              <Trash2 size={18} /> Geçmişi Sil
            </button>
          </div>
        </div>

        {/* Learned Words Section */}
        <div className={`flex-1 md:w-80 bg-white border-l border-gray-200 p-6 flex flex-col ${activeTab === 'learned' ? 'flex' : 'hidden md:flex'}`}>
          <h2 className="text-lg font-bold text-gray-800 mb-6 flex items-center gap-2">
            <BookOpen className="text-orange-600" /> Öğrendiklerim
          </h2>
          <div className="flex-1 overflow-y-auto space-y-3">
            {learnedWords.length === 0 ? (
              <p className="text-gray-400 text-center mt-10">Henüz bir kelime öğrenmedin. Koç ile çalışmaya başla!</p>
            ) : (
              learnedWords.map((word) => (
                <div key={word.id} className="bg-gray-50 p-4 rounded-xl border border-gray-100">
                  <p className="font-bold text-gray-800 text-lg">{word.original}</p>
                  <p className="text-gray-600">{word.translation}</p>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default EducationCoach;
