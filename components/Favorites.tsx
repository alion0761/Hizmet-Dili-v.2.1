import React from 'react';
import { ArchivedSession } from '../types';

interface FavoritesProps {
  favorites: ArchivedSession[];
  onOpen: (session: ArchivedSession) => void;
  t: (key: string, params?: Record<string, string>) => string;
}

const Favorites: React.FC<FavoritesProps> = ({ favorites, onOpen, t }) => {
  return (
    <div className="p-4 space-y-4">
      <h2 className="text-xl font-bold">{t('savedSessions')}</h2>
      {favorites.length === 0 ? (
        <p>{t('noRecordings')}</p>
      ) : (
        favorites.map((session) => (
          <div key={session.id} onClick={() => onOpen(session)} className="p-4 bg-slate-800 rounded-lg cursor-pointer">
            <p className="font-bold">{session.date}</p>
            <p className="text-sm text-slate-400">{session.preview}</p>
          </div>
        ))
      )}
    </div>
  );
};

export default Favorites;
