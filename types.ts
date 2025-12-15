export enum TargetLanguage {
  TURKISH = 'Turkish',
  ENGLISH = 'English',
  DUTCH = 'Dutch',
  ARABIC = 'Arabic',
  RUSSIAN = 'Russian',
  GERMAN = 'German',
  ITALIAN = 'Italian',
  FRENCH = 'French',
  UKRAINIAN = 'Ukrainian'
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'model';
  text: string;
  timestamp: Date;
  isFinal: boolean;
  langCode?: TargetLanguage; // Added to track which language this message is in
}

export interface AudioConfig {
  inputSampleRate: number;
  outputSampleRate: number;
}

export interface OfflinePack {
  id: string;
  pair: string; // e.g., "TR-EN"
  name: string;
  size: string;
  downloaded: boolean;
  progress: number; // 0-100
}