import React, { useState, useEffect, useRef } from 'react';
import { Waves, Mic, MicOff } from 'lucide-react';

interface LiveTranslationProps {
  sourceLang: string;
  targetLang: string;
}

const LiveTranslation: React.FC<LiveTranslationProps> = ({ sourceLang, targetLang }) => {
  const [isActive, setIsActive] = useState(false);
  const audioContextRef = useRef<AudioContext | null>(null);

  const startLiveMode = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;
      
      const source = audioContext.createMediaStreamSource(stream);
      const splitter = audioContext.createChannelSplitter(2);
      source.connect(splitter);

      // Here we would connect the split channels to our translation engine
      // For now, just setting active state
      setIsActive(true);
    } catch (error) {
      console.error('Error starting live mode:', error);
    }
  };

  const stopLiveMode = () => {
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    setIsActive(false);
  };

  return (
    <div className="flex-1 flex flex-col p-4 space-y-4">
      <div className="flex-1 grid grid-cols-2 gap-4">
        <div className="bg-slate-900/50 p-4 rounded-2xl border border-slate-800">
          <h3 className="text-slate-400 text-xs font-bold uppercase tracking-wider mb-2">{sourceLang} (Sol)</h3>
          <div className="text-slate-200">Konuşulanlar burada görünecek...</div>
        </div>
        <div className="bg-slate-900/50 p-4 rounded-2xl border border-slate-800">
          <h3 className="text-slate-400 text-xs font-bold uppercase tracking-wider mb-2">{targetLang} (Sağ)</h3>
          <div className="text-slate-200">Çeviriler burada görünecek...</div>
        </div>
      </div>
      <button 
        onClick={isActive ? stopLiveMode : startLiveMode}
        className={`w-full py-4 rounded-2xl flex items-center justify-center gap-2 font-bold ${isActive ? 'bg-red-600' : 'bg-blue-600'}`}
      >
        {isActive ? <MicOff size={20} /> : <Mic size={20} />}
        {isActive ? 'Canlı Modu Durdur' : 'Canlı Modu Başlat'}
      </button>
    </div>
  );
};

export default LiveTranslation;
