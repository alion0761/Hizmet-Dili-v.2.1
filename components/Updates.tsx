import React from 'react';
import { Sparkles, X } from 'lucide-react';

interface UpdatesProps {
  onClose: () => void;
  t: (key: string, params?: Record<string, string>) => string;
}

const Updates: React.FC<UpdatesProps> = ({ onClose, t }) => {
  return (
    <div className="fixed inset-0 z-[110] bg-black/80 backdrop-blur-md flex items-center justify-center p-6 animate-fade-in">
      <div className="bg-slate-900 border border-slate-800 w-full max-w-lg rounded-[2.5rem] shadow-2xl flex flex-col max-h-[80vh]">
        <div className="p-8 border-b border-slate-800 flex justify-between items-center">
          <h2 className="text-2xl font-bold flex items-center gap-3"><Sparkles className="text-emerald-400" /> {t('updatesTitle')}</h2>
          <button onClick={onClose} className="p-2 bg-slate-800 rounded-full hover:bg-slate-700 transition-colors"><X /></button>
        </div>
        <div className="p-8 overflow-y-auto space-y-8 no-scrollbar">
          <div className="relative pl-8 border-l-2 border-slate-800 space-y-2">
            <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-emerald-500 border-4 border-slate-900"></div>
            <div className="flex items-center gap-2">
              <span className="text-emerald-400 font-bold">v1.8</span>
              <span className="text-[10px] text-slate-500 uppercase tracking-widest">27 Mart 2026</span>
            </div>
            <h4 className="font-bold text-lg">Kapsamlı İyileştirmeler</h4>
            <ul className="text-sm text-slate-300 space-y-2 list-disc pl-4">
              <li>Karanlık mod desteği eklendi.</li>
              <li>Favoriler ve Çeviri Geçmişi özelliği getirildi.</li>
              <li>Kod yapısı modüler hale getirildi (Refactoring).</li>
              <li>Performans optimizasyonları yapıldı.</li>
            </ul>
          </div>
          {/* ... other versions ... */}
        </div>
      </div>
    </div>
  );
};

export default Updates;
