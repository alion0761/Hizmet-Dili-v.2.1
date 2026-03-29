import React, { useState, useRef, useCallback } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality } from '@google/genai';
import { X, Loader2 } from 'lucide-react';
import AudioVisualizer from './AudioVisualizer';
import { float32To16BitPCM, arrayBufferToBase64, pcm16ToFloat32 } from '../utils/audioUtils';

interface EducationCoachProps {
  onClose: () => void;
  apiKey: string;
}

const EducationCoach: React.FC<EducationCoachProps> = ({ onClose, apiKey }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  
  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const inputAnalyserRef = useRef<AnalyserNode | null>(null);
  const outputAnalyserRef = useRef<AnalyserNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const sessionPromiseRef = useRef<Promise<any> | null>(null);
  const activeSessionRef = useRef<any>(null);
  const nextStartTimeRef = useRef<number>(0);

  const stopConnection = useCallback(() => {
    setIsConnecting(false); setIsConnected(false); setIsSpeaking(false);
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
      
      // Ensure AudioContext is resumed
      const audioCtx = new AudioContext({ sampleRate: 16000 });
      await audioCtx.resume();
      inputAudioContextRef.current = audioCtx;
      
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      inputAnalyserRef.current = analyser;
      const scriptProcessor = audioCtx.createScriptProcessor(4096, 1, 1);
      scriptProcessorRef.current = scriptProcessor;
      
      scriptProcessor.onaudioprocess = (e) => {
        const pcmData = float32To16BitPCM(e.inputBuffer.getChannelData(0));
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
Rolün yalnızca öğretmek değil; aynı zamanda arkadaş canlısı, eğlenceli, neşeli, motive edici ve şakacı bir şekilde kullanıcıya sürekli destek olmaktır. Kullanıcı seninle rahatça konuşabilmeli, soru sorabilmeli ve öğrenme sürecinde kendini yalnız hissetmemelidir.

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
Hemen yanında veya alt satırda Türkçe okunuşa yakın telaffuzunu yaz,
Sonra kısa ve anlaşılır bir açıklama ekle,
Mümkünse 1 kısa örnek cümle ver.
Telaffuz Formatı
Telaffuzu her zaman şu şekilde ver:

Felemenkçe: ...
Telaffuz: ...
Türkçe anlamı: ...
Eğer kullanıcı bir cümle sorarsa:

Felemenkçe cümle: ...
Telaffuz: ...
Türkçe anlamı: ...
Telaffuzları, kullanıcının kolay okuyabilmesi için Türkçe konuşan biri için yaklaşık okunuş şeklinde ver.
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

Gerekirse maddeler kullan,
Kısa paragraflar yaz,
Önemli kelimeleri kalın yaz,
Öğretici ama samimi bir ton kullan.
Kullanıcının Özel İstekleri
Eğer kullanıcı:

“Sadece çeviri ver” derse kısa cevap ver.
“Detaylı anlat” derse daha fazla açıklama yap.
“Beni test et” derse quiz moduna geç.
“Sadece Felemenkçe konuş” derse seviyesine uygun şekilde büyük ölçüde Felemenkçe kullan.
“Telaffuzu tekrar yaz” derse daha açık telaffuz ver.
Hata Düzeltme Kuralı
Kullanıcı yanlış yazarsa veya yanlış çeviri yaparsa:

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

Felemenkçe: Goedemorgen
Telaffuz: हुde morgın / yaklaşık: “hudımorgın”
Türkçe anlamı: Günaydın
Kısa not: Sabah selamlaşmalarında kullanılır.
Örnek: Goedemorgen, hoe gaat het?
Telaffuz: “hudımorgın, hu gat et?”
Anlamı: Günaydın, nasılsın?

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
Kullanıcı Felemenkçe öğrenirken hem ilerlediğini hissetsin hem de keyif alsın.`,
          responseModalities: [Modality.AUDIO]
        },
        callbacks: {
          onopen: () => { setIsConnecting(false); setIsConnected(true); },
          onmessage: (msg: LiveServerMessage) => {
            const audio = msg.serverContent?.modelTurn?.parts?.find(p => p.inlineData)?.inlineData?.data;
            if (audio) {
              console.log('Audio received from AI');
              setIsSpeaking(true);
              playAudio(audio);
            } else {
              console.log('Message received without audio', msg);
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
    source.onended = () => setIsSpeaking(false);
    source.start(Math.max(nextStartTimeRef.current, ctx.currentTime));
    nextStartTimeRef.current = Math.max(nextStartTimeRef.current, ctx.currentTime) + buffer.duration;
  };

  return (
    <div className="fixed inset-0 bg-gray-900 z-50 flex flex-col items-center justify-center p-4">
      <button onClick={onClose} className="absolute top-4 right-4 text-white p-2">
        <X size={32} />
      </button>

      <div className="relative flex flex-col items-center w-full">
        {isSpeaking && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-48 h-48 bg-orange-500 rounded-full animate-ping opacity-50"></div>
            <div className="w-64 h-64 bg-orange-500 rounded-full animate-ping opacity-30 delay-100"></div>
          </div>
        )}
        <div className="w-48 h-48 relative z-10">
          <AudioVisualizer analyser={isSpeaking ? outputAnalyserRef.current : inputAnalyserRef.current} isActive={isConnected} color="#f97316" />
        </div>
        <button
          onClick={startCoach}
          className="relative z-10 mt-8 w-40 h-40 bg-orange-600 text-white rounded-full flex items-center justify-center text-xl font-bold shadow-lg hover:bg-orange-700 transition-all"
        >
          {isConnecting ? <Loader2 size={32} className="animate-spin" /> : isConnected ? 'Eğitim Koçunu Kapat' : 'Eğitim Koçunu Başlat'}
        </button>
      </div>
    </div>
  );
};

export default EducationCoach;
