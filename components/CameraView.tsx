import React, { useRef } from 'react';
import { Camera, X, Zap } from 'lucide-react';

interface CameraViewProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  onCapture: () => void;
  onClose: () => void;
  capturedImage: string | null;
  analysisResult: { translation: string, info: string } | null;
  isAnalyzing: boolean;
  t: (key: string, params?: Record<string, string>) => string;
}

const CameraView: React.FC<CameraViewProps> = ({ videoRef, canvasRef, onCapture, onClose, capturedImage, analysisResult, isAnalyzing, t }) => {
  return (
    <div className="fixed inset-0 bg-black z-50 flex flex-col">
      <div className="flex justify-between p-4 bg-slate-900">
        <button onClick={onClose}><X /></button>
        <h2 className="text-lg font-bold">{t('photoTranslation')}</h2>
        <div />
      </div>
      <div className="flex-1 relative">
        {!capturedImage ? (
          <video ref={videoRef} autoPlay playsInline className="w-full h-full object-cover" />
        ) : (
          <img src={capturedImage} className="w-full h-full object-contain" />
        )}
        <canvas ref={canvasRef} className="hidden" />
      </div>
      {!capturedImage && (
        <button onClick={onCapture} className="absolute bottom-8 left-1/2 -translate-x-1/2 p-4 bg-white rounded-full text-black">
          <Camera size={32} />
        </button>
      )}
      {analysisResult && (
        <div className="p-4 bg-slate-800 text-white">
          <p className="font-bold">{analysisResult.translation}</p>
          <p className="text-sm">{analysisResult.info}</p>
        </div>
      )}
      {isAnalyzing && <p className="p-4 text-center">{t('analyzingProduct')}</p>}
    </div>
  );
};

export default CameraView;
