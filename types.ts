export enum TargetLanguage {
  TURKISH = 'Turkish',
  ENGLISH = 'English',
  DUTCH = 'Dutch',
  ARABIC = 'Arabic',
  RUSSIAN = 'Russian',
  GERMAN = 'German',
  ITALIAN = 'Italian',
  FRENCH = 'French',
  UKRAINIAN = 'Ukrainian',
  PORTUGUESE = 'Portuguese',
  CHINESE = 'Chinese',
  SPANISH = 'Spanish',
  JAPANESE = 'Japanese'
}

export type UILanguage = 'tr' | 'en';

export enum AIProvider {
  GEMINI = 'Gemini',
  OPENAI = 'OpenAI',
  ANTHROPIC = 'Anthropic'
}

export interface APIKeys {
  gemini?: string;
  openai?: string;
  anthropic?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'model';
  text: string;
  timestamp: Date;
  isFinal: boolean;
  langCode?: TargetLanguage;
}

export interface AudioConfig {
  inputSampleRate: number;
  outputSampleRate: number;
}

export interface OfflinePack {
  id: string;
  pair: string;
  name: string;
  size: string;
  downloaded: boolean;
  progress: number;
}

export interface ArchivedSession {
  id: string;
  date: string;
  targetLang: TargetLanguage;
  preview: string;
  messages: ChatMessage[];
}