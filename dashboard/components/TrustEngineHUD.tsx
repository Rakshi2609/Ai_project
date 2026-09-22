"use client";

import React from "react";
import { Gauge, ShieldAlert, CheckCircle, AlertTriangle, Zap, Terminal, Activity } from "lucide-react";
import { InferenceResponse } from "@/types/telemetry";

interface TrustEngineHUDProps {
  inference: InferenceResponse | null;
  loading?: boolean;
}

export default function TrustEngineHUD({ inference, loading = false }: TrustEngineHUDProps) {
  const pred = inference?.module_4_trust_prediction;
  const policy = inference?.module_5_mitigation_policy;
  const attn = inference?.module_3_cross_modal_attention?.dynamic_attention_weights;

  const trustScore = pred?.continuous_trust_score ?? 0.73;
  const trustState = pred?.trust_state ?? "CALIBRATED_TRUST";
  const speedFactor = policy?.execution_speed_factor ?? 1.0;
  const actionName = policy?.recommended_action ?? "MAINTAIN_NOMINAL";
  const hudMessage = policy?.hud_transparency_message ?? "Cobot Status Nominal: Human-robot trust calibrated. Operating at full velocity.";
  const latencyMs = pred?.latency_ms ?? 1.1;

  // Arc Gauge offset
  const pct = Math.round(trustScore * 100);
  const strokeOffset = 100 - pct;

  // Color theme according to state
  const isUnderTrust = trustState === "UNDER_TRUST";
  const isOverTrust = trustState === "OVER_TRUST";
  const stateColor = isUnderTrust ? "text-rose-400" : isOverTrust ? "text-purple-400" : "text-emerald-400";
  const stateBg = isUnderTrust ? "bg-rose-500/20 border-rose-500/40 text-rose-300" : isOverTrust ? "bg-purple-500/20 border-purple-500/40 text-purple-300" : "bg-emerald-500/20 border-emerald-500/40 text-emerald-300";

  return (
    <div className="space-y-4">
      {/* Top 4 Metrics Strip */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        
        {/* 1. Continuous Trust Score */}
        <div className="glass-card p-4 flex flex-col justify-between hud-border border-teal-500/30">
          <div className="flex justify-between items-start">
            <span className="text-xs font-mono text-teal-300 font-medium flex items-center">
              <span className="w-1.5 h-1.5 rounded-full bg-teal-400 mr-1.5 animate-pulse" />
              Continuous Trust Score
            </span>
            <div className="p-1.5 rounded-lg bg-teal-500/10 border border-teal-500/20 text-teal-400">
              <Gauge className="w-4 h-4" />
            </div>
          </div>

          <div className="flex items-center space-x-3.5 my-2">
            <div className="relative w-16 h-16 flex items-center justify-center shrink-0">
              <svg className="w-full h-full -rotate-90 drop-shadow-[0_0_8px_rgba(45,212,191,0.5)]" viewBox="0 0 36 36">
                <path className="text-dark-800" strokeWidth="3.5" stroke="currentColor" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                <path
                  className={`gauge-circle transition-all duration-500 ${isUnderTrust ? "text-rose-500" : isOverTrust ? "text-purple-400" : "text-teal-400"}`}
                  strokeDasharray="100, 100"
                  strokeDashoffset={strokeOffset}
                  strokeWidth="3.5"
                  strokeLinecap="round"
                  stroke="currentColor"
                  fill="none"
                  d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                />
              </svg>
              <span className={`absolute text-xs font-black font-mono text-white ${isUnderTrust ? "animate-pulse text-rose-400" : ""}`}>
                {pct}%
              </span>
            </div>

            <div>
              <span
                className={`text-3xl font-black font-mono tracking-tight ${
                  isUnderTrust ? "text-rose-400 cyber-glow-rose" : "text-white cyber-glow-teal"
                }`}
              >
                {trustScore.toFixed(3)}
              </span>
              <p className={`text-[11px] font-semibold flex items-center mt-0.5 ${stateColor}`}>
                <span className="w-1.5 h-1.5 rounded-full bg-current mr-1.5 animate-ping" />
                {isUnderTrust ? "Disuse Risk (Plunging)" : isOverTrust ? "Misuse Risk (Complacent)" : "Optimal Calibrated"}
              </p>
            </div>
          </div>

          <div className="flex justify-between text-[10px] text-gray-400 font-mono pt-2 border-t border-white/5">
            <span>Range: 0.0 - 1.0</span>
            <span className="text-teal-400">Target: Calibrated</span>
          </div>
        </div>

        {/* 2. Categorical Trust State */}
        <div className={`glass-card p-4 flex flex-col justify-between hud-border ${isUnderTrust ? "border-rose-500/40 shadow-[0_0_20px_rgba(244,63,94,0.2)]" : "border-emerald-500/20"}`}>
          <div className="flex justify-between items-start">
            <span className="text-xs font-mono text-gray-300 font-medium">Trust State</span>
            <div className={`p-1.5 rounded-lg border ${stateBg}`}>
              {isUnderTrust ? <AlertTriangle className="w-4 h-4" /> : <CheckCircle className="w-4 h-4" />}
            </div>
          </div>

          <div className="my-2">
            <span className={`px-3 py-1.5 rounded-xl text-xs font-bold font-display tracking-wide inline-block border shadow-sm ${stateBg}`}>
              {trustState}
            </span>
          </div>
          <p className="text-[11px] text-gray-300 line-clamp-2 leading-relaxed">
            {pred?.state_description ?? "Real-time balanced cognitive and behavioral safety."}
          </p>
        </div>

        {/* 3. Closed-Loop Mitigation Policy */}
        <div className="glass-card p-4 flex flex-col justify-between hud-border border-cyan-500/30">
          <div className="flex justify-between items-start">
            <span className="text-xs font-mono text-cyan-300 font-medium">Mitigation Policy</span>
            <div className="p-1.5 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
              <ShieldAlert className="w-4 h-4" />
            </div>
          </div>

          <div className="my-2 flex items-center space-x-2.5">
            <span className="text-3xl font-black font-mono text-cyan-300 cyber-glow-teal">
              {speedFactor.toFixed(2)}x
            </span>
            <span className="text-xs font-bold text-gray-100 font-mono tracking-tight bg-dark-900 px-2 py-1 rounded border border-white/5">
              {actionName}
            </span>
          </div>
          <p className="text-[10px] text-cyan-400/90 font-mono line-clamp-1">
            {policy?.low_level_control_signal ?? "EXECUTE_NOMINAL_COOPERATIVE_CYCLE"}
          </p>
        </div>

        {/* 4. Real-Time Edge Latency */}
        <div className="glass-card p-4 flex flex-col justify-between hud-border border-amber-500/20">
          <div className="flex justify-between items-start">
            <span className="text-xs font-mono text-amber-300 font-medium">Decision Latency</span>
            <div className="p-1.5 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-400">
              <Zap className="w-4 h-4" />
            </div>
          </div>

          <div className="flex items-baseline space-x-2 my-2">
            <span className="text-3xl font-black font-mono text-white tracking-tight">
              {latencyMs.toFixed(1)} ms
            </span>
            <span className="text-xs text-emerald-400 font-mono font-bold">(&lt; 250ms target)</span>
          </div>
          <div className="flex items-center justify-between text-[11px] text-gray-400 pt-2 border-t border-white/5">
            <span>Confidence: <span className="text-teal-300 font-mono font-bold">94.2%</span></span>
            <span className="text-emerald-400 font-mono flex items-center">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1 animate-pulse" /> Edge Ready
            </span>
          </div>
        </div>

      </div>

      {/* Dynamic Cross-Modal Attention Weights */}
      <div className="glass-card p-4 space-y-3 border-teal-500/20">
        <div className="flex justify-between items-center text-xs font-mono border-b border-white/5 pb-2">
          <span className="font-bold text-white flex items-center font-display tracking-wide">
            <Activity className="w-4 h-4 mr-2 text-teal-400" />
            Cross-Modal Dynamic Attention Weights α(t)
          </span>
          <span className="text-emerald-400 text-[10px] font-semibold bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
            Softmax Gating Active
          </span>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs font-mono">
          <div>
            <div className="flex justify-between text-[11px] mb-1">
              <span className="text-blue-400 font-semibold">Robot Kinematics</span>
              <span className="text-white font-bold">{((attn?.robot_kinematics ?? 0.41) * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-dark-900 h-1.5 rounded-full overflow-hidden">
              <div
                className="bg-blue-500 h-full transition-all duration-300 shadow-[0_0_8px_#3b82f6]"
                style={{ width: `${(attn?.robot_kinematics ?? 0.41) * 100}%` }}
              />
            </div>
          </div>

          <div>
            <div className="flex justify-between text-[11px] mb-1">
              <span className="text-teal-300 font-semibold">Facial Affect</span>
              <span className="text-white font-bold">{((attn?.facial_affect ?? 0.25) * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-dark-900 h-1.5 rounded-full overflow-hidden">
              <div
                className="bg-teal-400 h-full transition-all duration-300 shadow-[0_0_8px_#2dd4bf]"
                style={{ width: `${(attn?.facial_affect ?? 0.25) * 100}%` }}
              />
            </div>
          </div>

          <div>
            <div className="flex justify-between text-[11px] mb-1">
              <span className="text-amber-400 font-semibold">Vocal Acoustics</span>
              <span className="text-white font-bold">{((attn?.vocal_acoustics ?? 0.14) * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-dark-900 h-1.5 rounded-full overflow-hidden">
              <div
                className="bg-amber-400 h-full transition-all duration-300 shadow-[0_0_8px_#fbbf24]"
                style={{ width: `${(attn?.vocal_acoustics ?? 0.14) * 100}%` }}
              />
            </div>
          </div>

          <div>
            <div className="flex justify-between text-[11px] mb-1">
              <span className="text-rose-400 font-semibold">Physio BVP/EDA</span>
              <span className="text-white font-bold">{((attn?.physiological_bvp ?? 0.20) * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-dark-900 h-1.5 rounded-full overflow-hidden">
              <div
                className="bg-rose-500 h-full transition-all duration-300 shadow-[0_0_8px_#f43f5e]"
                style={{ width: `${(attn?.physiological_bvp ?? 0.20) * 100}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Operator HUD Transparency Ticker */}
      <div className="glass-card p-3.5 flex items-center justify-between border-teal-500/20 bg-dark-950/80">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-teal-500/20 border border-teal-500/40 flex items-center justify-center text-teal-300 shrink-0">
            <Terminal className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-[11px] font-mono text-teal-300 font-medium">Cobot Transparency Feed:</span>
              <span className="text-[10px] font-mono text-gray-500">LIVE FEED</span>
            </div>
            <p className="text-xs text-gray-200 mt-0.5 font-sans leading-relaxed">{hudMessage}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
