"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import { Mic, MicOff, Volume2, Activity, Radio, AlertCircle } from "lucide-react";
import { VocalTelemetry } from "@/types/telemetry";

interface VoiceCaptureProps {
  onTelemetryChange?: (data: VocalTelemetry) => void;
  compact?: boolean;
}

export default function VoiceCapture({
  onTelemetryChange,
  compact = false,
}: VoiceCaptureProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const animFrameRef = useRef<number | null>(null);

  const [micActive, setMicActive] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [audioMetrics, setAudioMetrics] = useState<{
    f0_pitch: number;
    jitter: number;
    intensity_db: number;
    speaking: boolean;
  }>({
    f0_pitch: 165,
    jitter: 0.85,
    intensity_db: 54,
    speaking: false,
  });

  const lastPitchesRef = useRef<number[]>([]);

  // Autocorrelation Pitch (F0) Detector
  const autoCorrelate = (buffer: Float32Array, sampleRate: number): number => {
    let size = buffer.length;
    let sumOfSquares = 0;
    for (let i = 0; i < size; i++) {
      const val = buffer[i];
      sumOfSquares += val * val;
    }
    const rootMeanSquare = Math.sqrt(sumOfSquares / size);
    if (rootMeanSquare < 0.015) {
      return -1; // Not enough vocal energy
    }

    let r1 = 0, r2 = size - 1;
    const thres = 0.2;
    for (let i = 0; i < size / 2; i++) {
      if (Math.abs(buffer[i]) < thres) {
        r1 = i;
        break;
      }
    }
    for (let i = 1; i < size / 2; i++) {
      if (Math.abs(buffer[size - i]) < thres) {
        r2 = size - i;
        break;
      }
    }

    buffer = buffer.slice(r1, r2);
    size = buffer.length;

    const c = new Array(size).fill(0);
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size - i; j++) {
        c[i] = c[i] + buffer[j] * buffer[j + i];
      }
    }

    let d = 0;
    while (c[d] > c[d + 1]) d++;
    let maxval = -1, maxpos = -1;
    for (let i = d; i < size; i++) {
      if (c[i] > maxval) {
        maxval = c[i];
        maxpos = i;
      }
    }
    let T0 = maxpos;

    // Parabolic interpolation for fine tuning
    const x1 = c[T0 - 1], x2 = c[T0], x3 = c[T0 + 1];
    const a = (x1 + x3 - 2 * x2) / 2;
    const b = (x3 - x1) / 2;
    if (a) T0 = T0 - b / (2 * a);

    return sampleRate / T0;
  };

  // Start Real Microphone
  const startMic = async () => {
    setErrorMsg(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      mediaStreamRef.current = stream;

      const audioCtx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
      audioContextRef.current = audioCtx;

      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      analyserRef.current = analyser;

      const source = audioCtx.createMediaStreamSource(stream);
      source.connect(analyser);

      setMicActive(true);
    } catch (err: unknown) {
      console.warn("Microphone access error:", err);
      setErrorMsg("Microphone permission denied. Streaming simulated vocal acoustic model.");
      setMicActive(false);
    }
  };

  // Stop Microphone
  const stopMic = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    setMicActive(false);
  };

  // Visualization & Feature Extraction Loop
  const drawWaveform = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Dark grid lines
    ctx.strokeStyle = "rgba(51, 65, 85, 0.25)";
    ctx.lineWidth = 1;
    for (let y = 15; y < h; y += 15) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    if (micActive && analyserRef.current && audioContextRef.current) {
      const analyser = analyserRef.current;
      const bufferLength = analyser.fftSize;
      const timeData = new Float32Array(bufferLength);
      analyser.getFloatTimeDomainData(timeData);

      // Compute RMS Energy
      let sumSquares = 0;
      for (let i = 0; i < timeData.length; i++) {
        sumSquares += timeData[i] * timeData[i];
      }
      const rms = Math.sqrt(sumSquares / timeData.length);
      const intensityDb = Math.max(30, Math.min(95, 20 * Math.log10(rms + 1e-4) + 85));
      const isSpeaking = intensityDb > 45;

      // Compute Pitch
      const pitch = autoCorrelate(timeData, audioContextRef.current.sampleRate);
      let cleanPitch = audioMetrics.f0_pitch;
      if (pitch > 70 && pitch < 400) {
        cleanPitch = Math.round(pitch);
        lastPitchesRef.current.push(cleanPitch);
        if (lastPitchesRef.current.length > 10) lastPitchesRef.current.shift();
      }

      // Compute Acoustic Jitter
      let jitter = 0.85;
      if (lastPitchesRef.current.length > 2) {
        let diffSum = 0;
        for (let i = 1; i < lastPitchesRef.current.length; i++) {
          diffSum += Math.abs(lastPitchesRef.current[i] - lastPitchesRef.current[i - 1]);
        }
        jitter = parseFloat((diffSum / lastPitchesRef.current.length / 10).toFixed(2));
      }

      setAudioMetrics({
        f0_pitch: cleanPitch,
        jitter: Math.min(3.5, Math.max(0.3, jitter)),
        intensity_db: Math.round(intensityDb),
        speaking: isSpeaking,
      });

      if (onTelemetryChange) {
        onTelemetryChange({
          pitch_mean_hz: cleanPitch,
          acoustic_jitter_pct: Math.min(3.5, Math.max(0.3, jitter)),
          intensity_db: Math.round(intensityDb),
          pause_ratio: isSpeaking ? 0.05 : 0.45,
          speech_duration_s: isSpeaking ? 2.8 : 0.2,
        });
      }

      // Draw Glowing Oscilloscope Waveform
      ctx.save();
      const strokeColor = isSpeaking ? "#f59e0b" : "#2dd4bf";
      ctx.shadowBlur = 10;
      ctx.shadowColor = strokeColor;
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2.2;
      ctx.beginPath();

      const sliceWidth = (w * 1.0) / bufferLength;
      let x = 0;
      for (let i = 0; i < bufferLength; i += 4) {
        const v = timeData[i] * 1.8;
        const y = (h / 2) + (v * h / 2);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth * 4;
      }
      ctx.stroke();
      ctx.restore();
    } else {
      // Idle Simulated Sine Wave
      const time = Date.now() * 0.003;
      ctx.save();
      ctx.strokeStyle = "rgba(45, 212, 191, 0.4)";
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      for (let x = 0; x < w; x++) {
        const y = h / 2 + Math.sin(x * 0.05 + time) * 10;
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();
    }

    animFrameRef.current = requestAnimationFrame(drawWaveform);
  }, [micActive, audioMetrics.f0_pitch, onTelemetryChange]);

  useEffect(() => {
    animFrameRef.current = requestAnimationFrame(drawWaveform);
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      stopMic();
    };
  }, [drawWaveform]);

  return (
    <div className={`glass-card p-4 space-y-3 border-amber-500/30 ${compact ? "" : "shadow-xl"}`}>
      {/* Header */}
      <div className="flex items-center justify-between border-b border-white/5 pb-2.5">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-400">
            <Mic className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-xs font-bold text-white font-display tracking-wide flex items-center">
              Real Voice Prosody Capture
              <span className={`ml-2 w-2 h-2 rounded-full ${micActive ? "bg-amber-400 animate-pulse shadow-[0_0_8px_#fbbf24]" : "bg-gray-500"}`} />
            </h3>
            <p className="text-[10px] text-gray-400 font-mono">F0 Pitch, Acoustic Jitter &amp; Stress Metrics</p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          {micActive ? (
            <button
              onClick={stopMic}
              className="px-2.5 py-1 text-[11px] font-semibold rounded-lg bg-rose-500/20 text-rose-300 border border-rose-500/40 hover:bg-rose-500/30 transition flex items-center space-x-1"
            >
              <MicOff className="w-3.5 h-3.5" />
              <span>Mute Mic</span>
            </button>
          ) : (
            <button
              onClick={startMic}
              className="px-3 py-1 text-[11px] font-bold rounded-lg bg-gradient-to-r from-amber-400 to-orange-400 text-dark-950 hover:brightness-110 shadow-[0_0_12px_rgba(245,158,11,0.4)] transition flex items-center space-x-1 font-display"
            >
              <Radio className="w-3.5 h-3.5 text-dark-950" />
              <span>Enable Microphone</span>
            </button>
          )}
        </div>
      </div>

      {errorMsg && (
        <div className="p-2 rounded-lg bg-amber-500/15 border border-amber-500/30 text-amber-300 text-[11px] flex items-center space-x-2 font-mono">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Real-Time Oscilloscope Canvas */}
      <div className="relative h-28 rounded-xl overflow-hidden border border-white/10 bg-dark-950 shadow-inner">
        <canvas ref={canvasRef} width={400} height={112} className="w-full h-full" />
        <div className="absolute top-2 right-2 text-[9px] font-mono px-2 py-0.5 rounded bg-dark-900/85 border border-white/10 text-gray-300 flex items-center space-x-1">
          <Activity className="w-3 h-3 text-amber-400 animate-pulse" />
          <span>{micActive ? "LIVE SPEECH FFT" : "SYNTHETIC WAVE"}</span>
        </div>
      </div>

      {/* Extracted Vocal Metrics */}
      <div className="grid grid-cols-2 gap-2 text-xs font-mono">
        <div className="bg-dark-900/90 p-2.5 rounded-xl border border-white/5 space-y-1">
          <div className="flex justify-between items-center text-[10px]">
            <span className="text-gray-400">Pitch Frequency F0:</span>
            <span className="text-amber-300 font-bold text-sm">{audioMetrics.f0_pitch} Hz</span>
          </div>
          <div className="w-full bg-dark-800 rounded-full h-1.5 overflow-hidden">
            <div
              className="bg-amber-400 h-full transition-all duration-200"
              style={{ width: `${Math.min(100, (audioMetrics.f0_pitch / 300) * 100)}%` }}
            />
          </div>
        </div>

        <div className="bg-dark-900/90 p-2.5 rounded-xl border border-white/5 space-y-1">
          <div className="flex justify-between items-center text-[10px]">
            <span className="text-gray-400">Acoustic Jitter:</span>
            <span className={`font-bold text-sm ${audioMetrics.jitter > 1.5 ? "text-rose-400" : "text-emerald-400"}`}>
              {audioMetrics.jitter}%
            </span>
          </div>
          <div className="w-full bg-dark-800 rounded-full h-1.5 overflow-hidden">
            <div
              className={`h-full transition-all duration-200 ${audioMetrics.jitter > 1.5 ? "bg-rose-500" : "bg-emerald-400"}`}
              style={{ width: `${Math.min(100, (audioMetrics.jitter / 3.0) * 100)}%` }}
            />
          </div>
        </div>

        <div className="bg-dark-900/90 p-2 rounded-xl border border-white/5 flex justify-between items-center text-[11px]">
          <span className="text-gray-400 flex items-center"><Volume2 className="w-3.5 h-3.5 mr-1 text-teal-400" /> Sound Intensity:</span>
          <span className="text-teal-300 font-bold">{audioMetrics.intensity_db} dB</span>
        </div>

        <div className="bg-dark-900/90 p-2 rounded-xl border border-white/5 flex justify-between items-center text-[11px]">
          <span className="text-gray-400">Vocal State:</span>
          <span className={`font-bold ${audioMetrics.speaking ? "text-amber-400 animate-pulse" : "text-gray-400"}`}>
            {audioMetrics.speaking ? "ACTIVE SPEECH" : "QUIET / LISTENING"}
          </span>
        </div>
      </div>
    </div>
  );
}
