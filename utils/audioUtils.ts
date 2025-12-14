// Utility to convert Float32Array (Web Audio API default) to Int16Array (PCM 16-bit)
// and downsample if necessary.
export const float32To16BitPCM = (float32Arr: Float32Array): ArrayBuffer => {
  const buffer = new ArrayBuffer(float32Arr.length * 2);
  const view = new DataView(buffer);
  for (let i = 0; i < float32Arr.length; i++) {
    // Clamp the value between -1 and 1
    const s = Math.max(-1, Math.min(1, float32Arr[i]));
    // Convert to 16-bit integer (multiply by 32767)
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true); // true for little-endian
  }
  return buffer;
};

// Simple base64 encoder for ArrayBuffer
export const arrayBufferToBase64 = (buffer: ArrayBuffer): string => {
  let binary = '';
  const bytes = new Uint8Array(buffer);
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
};

// Base64 decoder to ArrayBuffer
export const base64ToArrayBuffer = (base64: string): ArrayBuffer => {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
};

// Convert Int16 PCM ArrayBuffer to Float32Array for Web Audio playback
export const pcm16ToFloat32 = (buffer: ArrayBuffer): Float32Array => {
  const int16Array = new Int16Array(buffer);
  const float32Array = new Float32Array(int16Array.length);
  for (let i = 0; i < int16Array.length; i++) {
    float32Array[i] = int16Array[i] / 32768.0;
  }
  return float32Array;
};
