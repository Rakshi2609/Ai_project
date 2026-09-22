"use client";

import React, { useState } from "react";
import FaceExpressionCapture from "@/components/FaceExpressionCapture";
import VoiceCapture from "@/components/VoiceCapture";
import { FacialTelemetry, VocalTelemetry } from "@/types/telemetry";
import { Camera, Radio, Activity, Eye, Smile, Frown, Sparkles, Volume2, Shield } from "lucide-react";

export default function CaptureStudioPage() {
  const [facialData, setFacialData] = useState<FacialTelemetry>({
    au04_brow_furrow: 0.12,
    blink_rate_bpm: 18,
    au12_smile: 0.08,
    mouth_open: 0.05,
    valence_entropy: 0.14,
  });

  const [vocalData, setVocalData] = useState<VocalTelemetry>({
    pitch_mean_hz: 168,
    acoustic_jitter_pct: 0.85,
    intensity_db: 54,
    pause_ratio: 0.15,
    speech_duration_s: 2.2,
  });

  // Derived Operator Cognitive Stress Index (0 - 100)
  const stressIndex = Math.min(
    100,
    Math.round(
      facialData.au04_brow_furrow * 45 +
        (vocalData.acoustic_jitter_pct / 3.0) * 35 +
        facialData.valence_entropy * 20
    )
  );

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="border-b border-white/5 pb-4">
        <h1 className="text-xl sm:text-2xl font-black text-white tracking-tight font-display flex items-center">
          <Camera className="w-6 h-6 mr-2.5 text-teal-400" />
          Operator Biometric Capture Station
        </h1>
        <p className="text-xs sm:text-sm text-gray-400 mt-1">
          High-resolution real-time facial expression tracking via WebRTC webcam and vocal prosody analysis via Web Audio API.
        </p>
      </div>

      {/* Top Biometric Stress Banner */}
      <div className="glass-card p-5 border-teal-500/30 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="flex items-center space-x-3.5">
          <div className="w-12 h-12 rounded-xl bg-teal-500/20 border border-teal-500/40 flex items-center justify-center text-teal-300">
            <Activity className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white font-display">
              Real-Time Human Cognitive Stress Index
            </h3>
            <p className="text-xs text-gray-400 font-mono mt-0.5">
              Multimodal Fusion of AU04 Brow Furrow + Acoustic Jitter + Micro-Expression Entropy
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <div className="text-right font-mono">
            <span className="text-[10px] text-gray-400 block">Calculated Stress</span>
            <span className={`text-2xl font-black ${stressIndex > 50 ? "text-rose-400" : "text-emerald-400"}`}>
              {stressIndex} / 100
            </span>
          </div>
          <span
            className={`px-3 py-1.5 rounded-xl text-xs font-bold uppercase font-mono border ${
              stressIndex > 50
                ? "bg-rose-500/20 text-rose-300 border-rose-500/40 shadow-[0_0_12px_rgba(244,63,94,0.3)]"
                : "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
            }`}
          >
            {stressIndex > 50 ? "Elevated Stress" : "Calm / Nominal"}
          </span>
        </div>
      </div>

      {/* Grid: Live Face on Left, Live Voice on Right */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {/* Real Face Expression Capture */}
        <div>
          <FaceExpressionCapture
            onTelemetryChange={(data) => setFacialData(data)}
          />
        </div>

        {/* Real Voice Capture */}
        <div>
          <VoiceCapture
            onTelemetryChange={(data) => setVocalData(data)}
          />
        </div>

      </div>

      {/* Raw Multimodal Stream Inspector */}
      <div className="glass-card p-4 space-y-2 border-white/10 bg-dark-950/80">
        <div className="flex items-center justify-between border-b border-white/5 pb-2 text-xs font-mono">
          <span className="text-gray-300 font-bold flex items-center">
            <Shield className="w-4 h-4 mr-1.5 text-teal-400" />
            Live Biometric Telemetry Stream Packet
          </span>
          <span className="text-emerald-400 text-[10px]">JSON Payloads Streamed to /api/infer</span>
        </div>
        <pre className="text-[11px] font-mono text-cyan-300 bg-dark-900 p-3 rounded-xl overflow-x-auto border border-white/5 max-h-48">
          {JSON.stringify(
            {
              timestamp_utc: new Date().toISOString(),
              facial_params: facialData,
              vocal_params: vocalData,
              derived_stress_index: stressIndex,
            },
            null,
            2
          )}
        </pre>
      </div>
    </div>
  );
}
