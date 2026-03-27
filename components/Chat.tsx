import React from 'react';
import { ChatMessage, TargetLanguage } from '../types';

interface ChatProps {
  messages: ChatMessage[];
  t: (key: string, params?: Record<string, string>) => string;
  handlePlaySpeech: (text: string, messageId: string) => void;
  isSpeechPlaying: string | null;
}

const Chat: React.FC<ChatProps> = ({ messages, t, handlePlaySpeech, isSpeechPlaying }) => {
  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {messages.map((msg) => (
        <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
          <div className={`max-w-[80%] p-3 rounded-lg ${msg.role === 'user' ? 'bg-blue-600' : 'bg-slate-800'}`}>
            <p>{msg.text}</p>
            {msg.role === 'model' && (
              <button onClick={() => handlePlaySpeech(msg.text, msg.id)} className="mt-2 text-xs text-slate-400">
                {isSpeechPlaying === msg.id ? t('listeningWithLang', { lang: msg.langCode || '' }) : t('listenAction')}
              </button>
            )}
          </div>
        </div>
      ))}
    </div>
  );
};

export default Chat;
