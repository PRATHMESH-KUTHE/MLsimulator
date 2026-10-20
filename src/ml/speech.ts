/**
 * Text-to-Speech (TTS) SpeechSynthesis helper for educational audio narration
 */

export interface SpeechController {
  speak: (text: string, onEnd?: () => void) => void;
  pause: () => void;
  resume: () => void;
  stop: () => void;
  isSpeaking: () => boolean;
  isPaused: () => boolean;
  isSupported: () => boolean;
}

export function createSpeechController(): SpeechController {
  const isSupported = () => typeof window !== 'undefined' && 'speechSynthesis' in window;

  const speak = (text: string, onEnd?: () => void) => {
    if (!isSupported()) return;
    window.speechSynthesis.cancel(); // Cancel any prior speech
    const cleanText = text.replace(/\\mathbf\{([^}]+)\}/g, '$1').replace(/\\frac\{([^}]+)\}\{([^}]+)\}/g, '$1 over $2').replace(/[\\_^{}]/g, '');
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    if (onEnd) {
      utterance.onend = onEnd;
      utterance.onerror = onEnd;
    }
    window.speechSynthesis.speak(utterance);
  };

  const pause = () => {
    if (isSupported()) window.speechSynthesis.pause();
  };

  const resume = () => {
    if (isSupported()) window.speechSynthesis.resume();
  };

  const stop = () => {
    if (isSupported()) window.speechSynthesis.cancel();
  };

  const isSpeaking = () => {
    return isSupported() ? window.speechSynthesis.speaking : false;
  };

  const isPaused = () => {
    return isSupported() ? window.speechSynthesis.paused : false;
  };

  return {
    speak,
    pause,
    resume,
    stop,
    isSpeaking,
    isPaused,
    isSupported,
  };
}
