"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import Cobot3DView from "@/components/Cobot3DView";
import FaceExpressionCapture from "@/components/FaceExpressionCapture";
import VoiceCapture from "@/components/VoiceCapture";
import TrustEngineHUD from "@/components/TrustEngineHUD";
import { FacialTelemetry, VocalTelemetry, RobotState, InferenceResponse } from "@/types/telemetry";
import { Sparkles, Play, RotateCcw, AlertTriangle, ShieldCheck } from "lucide-react";

export default function MasterCockpitPage() {
  // Multimodal Telemetry States
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

  const [robotState, setRobotState] = useState<RobotState>({
    mode: "correct",
    error_type: "nominal",
    speed_mps: 0.8,
    drift_m: 0.02,
    torque_anomaly: 0.04,
  });

  // Inference Prediction Output
  const [inference, setInference] = useState<InferenceResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const isInferringRef = useRef<boolean>(false);
  const hasPendingCallRef = useRef<boolean>(false);

  // Send Multimodal Packet to FastAPI Backend with guaranteed delivery
  const runInference = useCallback(async () => {
    if (isInferringRef.current) {
      hasPendingCallRef.current = true;
      return;
    }

    try {
      isInferringRef.current = true;
      setLoading(true);

      const payload = {
        task_name: "UR5 High-Precision Collaborative Assembly",
        planned_trajectory: [[0, 0, 0], [0.1, 0.2, 0.3], [0.2, 0.4, 0.1]],
        actual_trajectory: [
          [0, 0, 0],
          [0.1 + robotState.drift_m, 0.2, 0.3],
          [0.2 + robotState.drift_m, 0.4, 0.1],
        ],
        execution_speed_mps: robotState.speed_mps,
        error_type: robotState.error_type,
        joint_torque_anomaly: robotState.torque_anomaly,
        facial_params: {
          ...facialData,
          brow_furrow: facialData.au04_brow_furrow,
          fear_expression: Math.max(0, facialData.au04_brow_furrow * 0.85 + facialData.mouth_open * 0.45 - facialData.au12_smile * 0.4),
          anger_expression: Math.max(0, facialData.au04_brow_furrow * 0.80 - facialData.au12_smile * 0.5),
          eye_widen: Math.min(1.0, facialData.mouth_open * 0.7 + facialData.au04_brow_furrow * 0.3),
          jaw_clench: Math.max(0, 0.05 + facialData.au04_brow_furrow * 0.5 - facialData.mouth_open * 0.2),
          facial_entropy: facialData.valence_entropy,
          gaze_drift_variance: 0.04,
          dominant_emotion: facialData.au04_brow_furrow > 0.35 ? "Stressed (AU04)" : (facialData.au12_smile > 0.35 ? "Smiling (Positive)" : "Neutral")
        },
        vocal_params: {
          ...vocalData,
          pitch_f0_hz: vocalData.pitch_mean_hz,
          f0_std_hz: Math.max(8.0, vocalData.acoustic_jitter_pct * 16.0),
          jitter_percent: vocalData.acoustic_jitter_pct,
          shimmer_percent: Math.max(1.0, Math.min(12.0, (vocalData.intensity_db - 30.0) / 4.5)),
          pause_ratio: vocalData.pause_ratio,
          ambient_noise_snr_db: Math.max(10.0, 45.0 - (vocalData.intensity_db / 3.0)),
        },
        physio_params: {
          heart_rate_bpm: Math.round(72 + facialData.au04_brow_furrow * 35 + (vocalData.intensity_db > 65 ? 14 : 0)),
          hrv_rmssd_ms: Math.round(Math.max(18, 65 - facialData.au04_brow_furrow * 45)),
          eda_microsiemens: parseFloat((2.0 + facialData.au04_brow_furrow * 6.0 + vocalData.acoustic_jitter_pct * 1.4).toFixed(2)),
          has_motion_artifact: false,
          artifact_snr_db: 25.0,
        },
      };

      const res = await fetch("/api/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        const data: InferenceResponse = await res.json();
        setInference(data);
      }
    } catch (err) {
      console.error("Live inference error:", err);
    } finally {
      setLoading(false);
      isInferringRef.current = false;
      if (hasPendingCallRef.current) {
        hasPendingCallRef.current = false;
        runInference();
      }
    }
  }, [facialData, vocalData, robotState]);

  // Trigger inference on state changes
  useEffect(() => {
    runInference();
  }, [runInference]);

  // Periodic refresh loop every 1.5s to ensure continuous stream sync
  useEffect(() => {
    const timer = setInterval(() => {
      runInference();
    }, 1500);
    return () => clearInterval(timer);
  }, [runInference]);

  return (
    <div className="space-y-6">
      {/* Page Title & Status */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 border-b border-white/5 pb-4">
        <div>
          <h1 className="text-xl sm:text-2xl font-black text-white tracking-tight font-display flex items-center">
            Real-Time Multimodal Cockpit
            <span className="ml-3 px-2.5 py-0.5 rounded-full text-xs font-mono font-medium bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 flex items-center">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5 animate-pulse" />
              Live Edge Fusion
            </span>
          </h1>
          <p className="text-xs sm:text-sm text-gray-400 mt-1">
            Real facial expression capture, real voice prosody tracking, and 3D UR5 cobot with correct vs wrong fault injection.
          </p>
        </div>

        {/* Quick Scenario Preset Pills */}
        <div className="flex items-center space-x-1.5 overflow-x-auto py-1">
          <button
            onClick={() => {
              setRobotState({ mode: "correct", error_type: "nominal", speed_mps: 0.8, drift_m: 0.02, torque_anomaly: 0.04 });
              setFacialData({ au04_brow_furrow: 0.08, blink_rate_bpm: 16, au12_smile: 0.25, mouth_open: 0.02, valence_entropy: 0.1 });
            }}
            className="px-3 py-1.5 text-xs font-semibold rounded-xl bg-dark-800 hover:bg-dark-750 text-gray-200 border border-white/10 transition flex items-center space-x-1.5 shadow-sm"
          >
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            <span>Nominal</span>
          </button>

          <button
            onClick={() => {
              setRobotState({ mode: "wrong", error_type: "gripper_slip", speed_mps: 0.2, drift_m: 0.15, torque_anomaly: 0.25 });
              setFacialData({ au04_brow_furrow: 0.85, blink_rate_bpm: 38, au12_smile: 0.0, mouth_open: 0.45, valence_entropy: 0.8 });
            }}
            className="px-3 py-1.5 text-xs font-semibold rounded-xl bg-dark-800 hover:bg-dark-750 text-gray-200 border border-white/10 transition flex items-center space-x-1.5 shadow-sm"
          >
            <span className="w-2 h-2 rounded-full bg-rose-500 animate-ping" />
            <span>Slip Error</span>
          </button>
        </div>
      </div>

      {/* Top Section: Trust Engine Prediction HUD */}
      <TrustEngineHUD inference={inference} loading={loading} />

      {/* Middle Section: 3D Cobot on Left, Live Face & Voice Capture on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* 3D UR5 Cobot Digital Twin (Cols 1-7) */}
        <div className="lg:col-span-7">
          <Cobot3DView
            onRobotStateChange={(st) => setRobotState(st)}
            trustState={inference?.module_4_trust_prediction?.trust_state}
          />
        </div>

        {/* Live Real Biometrics (Face & Audio) (Cols 8-12) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Live Webcam Face Expression Capture */}
          <FaceExpressionCapture
            compact
            onTelemetryChange={(telemetry) => setFacialData(telemetry)}
          />

          {/* Live Microphone Voice Prosody Capture */}
          <VoiceCapture
            compact
            onTelemetryChange={(telemetry) => setVocalData(telemetry)}
          />
        </div>

      </div>
    </div>
  );
}
