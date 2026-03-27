import React from 'react';
import { AIProvider, UILanguage, TranslationContext } from '../types';
import { Settings as SettingsIcon, X, History, BookOpen, LogOut, User } from 'lucide-react';

interface SettingsProps {
  apiKeys: any;
  setApiKeys: (keys: any) => void;
  selectedProvider: AIProvider;
  setSelectedProvider: (provider: AIProvider) => void;
  uiLanguage: UILanguage;
  setUiLanguage: (lang: UILanguage) => void;
  voiceType: 'female' | 'male';
  setVoiceType: (type: 'female' | 'male') => void;
  translationContext: TranslationContext;
  setTranslationContext: (context: TranslationContext) => void;
  onClose: () => void;
  showUpdates: () => void;
  showGuide: () => void;
  t: (key: string, params?: Record<string, string>) => string;
}

const Settings: React.FC<SettingsProps> = ({ apiKeys, setApiKeys, selectedProvider, setSelectedProvider, uiLanguage, setUiLanguage, voiceType, setVoiceType, translationContext, setTranslationContext, onClose, showUpdates, showGuide, t }) => {
  return (
    <div className="fixed inset-0 z-[100] bg-black/80 backdrop-blur-md flex items-center justify-center p-6 animate-fade-in">
      <div className="bg-slate-900 border border-slate-800 w-full max-w-lg rounded-[2.5rem] shadow-2xl flex flex-col max-h-[80vh]">
        <div className="p-8 border-b border-slate-800 flex justify-between items-center">
          <h2 className="text-2xl font-bold flex items-center gap-3"><SettingsIcon size={24} className="text-blue-400" /> {t('settings')}</h2>
          <button onClick={onClose} className="p-2 bg-slate-800 rounded-full hover:bg-slate-700 transition-colors"><X /></button>
        </div>
        <div className="p-8 overflow-y-auto space-y-8 no-scrollbar">
          <div className="space-y-4">
            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest px-1">{t('context')}</h3>
            <div className="grid grid-cols-2 gap-2">
              {Object.values(TranslationContext).map((ctx) => (
                <button 
                  key={ctx} 
                  onClick={() => setTranslationContext(ctx)}
                  className={`py-2 rounded-lg font-bold text-xs transition-all border ${translationContext === ctx ? 'bg-blue-600 border-blue-500 text-white shadow-lg' : 'bg-slate-800 border-slate-700 text-slate-400'}`}
                >
                  {t(`context${ctx}`)}
                </button>
              ))}
            </div>
          </div>
          
          <div className="space-y-4">
            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest px-1">{t('aiModel')}</h3>
            <div className="grid grid-cols-3 gap-2">
              {[AIProvider.GEMINI, AIProvider.OPENAI, AIProvider.ANTHROPIC].map((p) => (
                <button 
                  key={p} 
                  onClick={() => { setSelectedProvider(p); localStorage.setItem('selected_provider', p); }}
                  className={`py-4 rounded-2xl font-bold text-xs transition-all border ${selectedProvider === p ? 'bg-blue-600 border-blue-500 text-white shadow-lg' : 'bg-slate-900 border-slate-800 text-slate-400'}`}
                >
                  {p}
                </button>
              ))}
            </div>
            <div className="space-y-2">
              <label className="text-xs font-bold text-slate-500 uppercase tracking-widest px-1">{t('aiKeyLabel', { provider: selectedProvider })}</label>
              <input 
                type="password" 
                value={apiKeys[selectedProvider.toLowerCase()] || ''}
                onChange={(e) => {
                  const newKeys = { ...apiKeys, [selectedProvider.toLowerCase()]: e.target.value };
                  setApiKeys(newKeys);
                  localStorage.setItem('ai_api_keys', JSON.stringify(newKeys));
                }}
                placeholder={t('apiKeyPlaceholder', { provider: selectedProvider })}
                className="w-full p-4 bg-slate-800 rounded-2xl border border-slate-700 focus:outline-none focus:border-blue-500 transition-colors"
              />
            </div>
          </div>

          <div className="space-y-4">
            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest px-1">{t('voicePreference')}</h3>
            <div className="grid grid-cols-2 gap-3">
              <button 
                onClick={() => setVoiceType('female')}
                className={`py-4 rounded-2xl flex flex-col items-center justify-center gap-2 transition-all border ${voiceType === 'female' ? 'bg-blue-600 border-blue-500 text-white shadow-lg' : 'bg-slate-900 border-slate-800 text-slate-400'}`}
              >
                  <span className="text-2xl">👩</span>
                  <span className="font-bold text-xs">{t('female')}</span>
              </button>
              <button 
                onClick={() => setVoiceType('male')}
                className={`py-4 rounded-2xl flex flex-col items-center justify-center gap-2 transition-all border ${voiceType === 'male' ? 'bg-blue-600 border-blue-500 text-white shadow-lg' : 'bg-slate-900 border-slate-800 text-slate-400'}`}
              >
                  <span className="text-2xl">👨</span>
                  <span className="font-bold text-xs">{t('male')}</span>
              </button>
            </div>
            <p className="text-[10px] text-slate-500 px-1 italic">{t('voiceNote')}</p>
          </div>

          <div className="space-y-4">
            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest px-1">{t('info')}</h3>
            <div className="grid grid-cols-2 gap-3">
              <button onClick={showUpdates} className="p-5 bg-slate-800/50 hover:bg-slate-800 border border-slate-700/50 rounded-2xl flex flex-col items-center gap-2 transition-all group">
                <div className="p-3 bg-blue-600/20 rounded-xl group-hover:bg-blue-600/40 transition-colors"><History size={20} className="text-blue-400" /></div>
                <span className="text-xs font-bold">{t('updates')}</span>
              </button>
              <button onClick={showGuide} className="p-5 bg-slate-800/50 hover:bg-slate-800 border border-slate-700/50 rounded-2xl flex flex-col items-center gap-2 transition-all group">
                <div className="p-3 bg-emerald-600/20 rounded-xl group-hover:bg-emerald-600/40 transition-colors"><BookOpen size={20} className="text-emerald-400" /></div>
                <span className="text-xs font-bold">{t('guideTitle')}</span>
              </button>
            </div>
          </div>

          <div className="space-y-4">
            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest px-1">{t('security')}</h3>
            <button 
              onClick={() => { localStorage.removeItem('ai_api_keys'); localStorage.removeItem('gemini_api_key'); window.location.reload(); }} 
              className="w-full p-5 bg-red-500/10 hover:bg-red-500/20 text-red-400 rounded-2xl font-bold flex items-center justify-center gap-2 transition-colors border border-red-500/10"
            >
              <LogOut size={20} /> {t('clearData')}
            </button>
          </div>
          
          <div className="text-center text-[10px] text-slate-600 pt-4 uppercase tracking-[0.2em]">{t('stableBuild')}</div>
        </div>
      </div>
    </div>
  );
};

export default Settings;
