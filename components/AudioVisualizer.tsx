import React, { useEffect, useRef } from 'react';

interface AudioVisualizerProps {
  analyser: AnalyserNode | null;
  isActive: boolean;
  color?: string;
}

const AudioVisualizer: React.FC<AudioVisualizerProps> = ({ analyser, isActive, color = '#4ade80' }) => { // Default to green-400 equivalent
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current || !analyser || !isActive) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size to match display size for sharpness
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    let animationId: number;

    const draw = () => {
      animationId = requestAnimationFrame(draw);
      analyser.getByteFrequencyData(dataArray);

      const width = rect.width;
      const height = rect.height;
      const centerY = height / 2;

      ctx.clearRect(0, 0, width, height);

      if (!isActive) return;

      // Modern visualizer settings
      const barCount = 30; // Fewer bars for a cleaner look
      const step = Math.floor(bufferLength / barCount); 
      const barWidth = (width / barCount) / 1.5; // Spacing
      const cornerRadius = barWidth / 2;

      // Create Gradient
      const gradient = ctx.createLinearGradient(0, 0, 0, height);
      gradient.addColorStop(0, `${color}00`); // Transparent at top
      gradient.addColorStop(0.5, color);       // Solid color in middle
      gradient.addColorStop(1, `${color}00`);  // Transparent at bottom

      ctx.fillStyle = gradient;

      for (let i = 0; i < barCount; i++) {
        // Average out a chunk of frequencies for smoother visualization
        let value = 0;
        for (let j = 0; j < step; j++) {
            value += dataArray[(i * step) + j];
        }
        value = value / step;

        // Scale value to height
        const barHeight = Math.max(4, (value / 255) * height * 0.8); 
        const x = (width - (barCount * barWidth * 1.5)) / 2 + (i * barWidth * 1.5);
        const y = centerY - (barHeight / 2);

        // Draw Rounded Bar
        ctx.beginPath();
        ctx.roundRect(x, y, barWidth, barHeight, cornerRadius);
        ctx.fill();
      }
    };

    draw();

    return () => {
      cancelAnimationFrame(animationId);
      if(ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
    };
  }, [analyser, isActive, color]);

  return (
    <canvas 
      ref={canvasRef} 
      className="w-full h-full"
      style={{ width: '100%', height: '100%' }}
    />
  );
};

export default AudioVisualizer;