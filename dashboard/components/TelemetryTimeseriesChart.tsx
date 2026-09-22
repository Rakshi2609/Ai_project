"use client";

import React, { useState, useEffect, useRef } from "react";
import { Activity, Clock, TrendingUp, Filter, Eye, EyeOff, Sparkles } from "lucide-react";
import { FacialTelemetry, VocalTelemetry, RobotState, InferenceResponse } from "@/types/telemetry";

interface TelemetryPoint {
  timestamp: number;
  trustScore: number;       // 0 - 100%
  au04BrowFurrow: number;   // 0 - 100%
  au12Smile: number;        // 0 - 100%
  acousticTension: number;  // 0 - 100%
  robotDrift: number;       // 0 - 100% (scaled to 25mm max)
  robotSpeed: number;       // 0 - 100% (scaled to 1.0 m/s max)
  heartRate: number;        // bpm
}

interface TelemetryTimeseriesChartProps {
  inference: InferenceResponse | null;
  facialData?: FacialTelemetry;
  vocalData?: VocalTelemetry;
  robotState?: RobotState;
}

export default function TelemetryTimeseriesChart({
  inference,
  facialData,
  vocalData,
  robotState,
}: TelemetryTimeseriesChartProps) {
  const [history, setHistory] = useState<TelemetryPoint[]>([]);
  const historyRef = useRef<TelemetryPoint[]>([]);
  
  // Modality Line Visibility Toggles
  const [visibleLines, setVisibleLines] = useState<{
    trust: boolean;
    au04: boolean;
    au12: boolean;
    tension: boolean;
    drift: boolean;
  }>({
    trust: true,
    au04: true,
    au12: true,
    tension: true,
    drift: true,
  });

  const toggleLine = (key: keyof typeof visibleLines) => {
    setVisibleLines((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  // Record telemetry sample every 200ms into rolling 15-second window
  useEffect(() => {
    const interval = setInterval(() => {
      const now = Date.now();
      const cutoff = now - 15000; // strictly last 15 seconds

      // Extract current values with sensible fallbacks
      const currentTrust = inference?.module_4_trust_prediction?.continuous_trust_score !== undefined
        ? inference.module_4_trust_prediction.continuous_trust_score * 100
        : 72;

      const currentAU04 = facialData?.au04_brow_furrow !== undefined
        ? facialData.au04_brow_furrow * 100
        : 12;

      const currentAU12 = facialData?.au12_smile !== undefined
        ? facialData.au12_smile * 100
        : 10;

      const jitter = vocalData?.acoustic_jitter_pct || 0.85;
      const currentTension = Math.min(100, Math.max(5, (jitter / 5.0) * 100));

      const driftMm = (robotState?.drift_m || 0.02) * 1000;
      const currentDrift = Math.min(100, Math.max(0, (driftMm / 25.0) * 100));

      const speedMps = robotState?.speed_mps || 0.8;
      const currentSpeed = Math.min(100, Math.max(0, (speedMps / 1.0) * 100));

      const currentHR = Math.round(72 + (currentAU04 / 100) * 32);

      const newPoint: TelemetryPoint = {
        timestamp: now,
        trustScore: Math.round(currentTrust),
        au04BrowFurrow: Math.round(currentAU04),
        au12Smile: Math.round(currentAU12),
        acousticTension: Math.round(currentTension),
        robotDrift: Math.round(currentDrift),
        robotSpeed: Math.round(currentSpeed),
        heartRate: currentHR,
      };

      // Append and filter out samples older than 15s
      const updated = [...historyRef.current, newPoint].filter((p) => p.timestamp >= cutoff);
      historyRef.current = updated;
      setHistory(updated);
    }, 200);

    return () => clearInterval(interval);
  }, [inference, facialData, vocalData, robotState]);

  // SVG Chart Geometry Constants
  const width = 800;
  const height = 220;
  const padLeft = 40;
  const padRight = 20;
  const padTop = 15;
  const padBottom = 28;

  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;

  // Build SVG Path from series
  const buildSvgPath = (extractor: (p: TelemetryPoint) => number) => {
    if (history.length < 2) return "";
    const now = Date.now();
    const windowMs = 15000;

    return history
      .map((p, idx) => {
        const timeOffset = p.timestamp - (now - windowMs);
        const x = padLeft + (Math.max(0, Math.min(windowMs, timeOffset)) / windowMs) * chartW;
        const yVal = Math.max(0, Math.min(100, extractor(p)));
        const y = padTop + chartH - (yVal / 100) * chartH;
        return `${idx === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(" ");
  };

  // Statistics over the last 15s
  const latestPoint = history[history.length - 1] || {
    trustScore: 72,
    au04BrowFurrow: 12,
    au12Smile: 10,
    acousticTension: 15,
    robotDrift: 8,
  };

  const firstPoint = history[0] || latestPoint;
  const trustDelta = latestPoint.trustScore - firstPoint.trustScore;

  return (
    <div className="glass-card p-4 space-y-3 border-teal-500/30">
      {/* Header & Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-white/5 pb-2.5">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-lg bg-teal-500/10 border border-teal-500/20 text-teal-300">
            <Activity className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-xs font-bold text-white font-display tracking-wide flex items-center">
              15-Second Multi-Parameter Rolling Dynamics
              <span className="ml-2 inline-flex items-center px-1.5 py-0.5 rounded text-[9px] font-mono bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5 animate-pulse" />
                Live 5 Hz Buffer
              </span>
            </h3>
            <p className="text-[10px] text-gray-400 font-mono">
              Temporal window [t - 15s, t] mapping trust calibration against affective &amp; mechanical shifts
            </p>
          </div>
        </div>

        {/* Visibility Toggles */}
        <div className="flex flex-wrap items-center gap-1.5 text-[10px] font-mono">
          {/* Continuous Trust */}
          <button
            onClick={() => toggleLine("trust")}
            className={`px-2 py-1 rounded-md border flex items-center space-x-1 transition ${
              visibleLines.trust
                ? "bg-teal-500/20 text-teal-300 border-teal-500/40 font-bold"
                : "bg-dark-900 text-gray-500 border-white/5 line-through opacity-60"
            }`}
          >
            <span className="w-2 h-2 rounded-full bg-teal-400 mr-1" />
            <span>Trust: {latestPoint.trustScore}%</span>
          </button>

          {/* AU04 Brow Furrow */}
          <button
            onClick={() => toggleLine("au04")}
            className={`px-2 py-1 rounded-md border flex items-center space-x-1 transition ${
              visibleLines.au04
                ? "bg-rose-500/20 text-rose-300 border-rose-500/40 font-bold"
                : "bg-dark-900 text-gray-500 border-white/5 line-through opacity-60"
            }`}
          >
            <span className="w-2 h-2 rounded-full bg-rose-400 mr-1" />
            <span>AU04 Brow: {latestPoint.au04BrowFurrow}%</span>
          </button>

          {/* AU12 Smile */}
          <button
            onClick={() => toggleLine("au12")}
            className={`px-2 py-1 rounded-md border flex items-center space-x-1 transition ${
              visibleLines.au12
                ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40 font-bold"
                : "bg-dark-900 text-gray-500 border-white/5 line-through opacity-60"
            }`}
          >
            <span className="w-2 h-2 rounded-full bg-emerald-400 mr-1" />
            <span>AU12 Smile: {latestPoint.au12Smile}%</span>
          </button>

          {/* Vocal Tension */}
          <button
            onClick={() => toggleLine("tension")}
            className={`px-2 py-1 rounded-md border flex items-center space-x-1 transition ${
              visibleLines.tension
                ? "bg-amber-500/20 text-amber-300 border-amber-500/40 font-bold"
                : "bg-dark-900 text-gray-500 border-white/5 line-through opacity-60"
            }`}
          >
            <span className="w-2 h-2 rounded-full bg-amber-400 mr-1" />
            <span>Voice Tension: {latestPoint.acousticTension}%</span>
          </button>

          {/* Cobot Drift */}
          <button
            onClick={() => toggleLine("drift")}
            className={`px-2 py-1 rounded-md border flex items-center space-x-1 transition ${
              visibleLines.drift
                ? "bg-purple-500/20 text-purple-300 border-purple-500/40 font-bold"
                : "bg-dark-900 text-gray-500 border-white/5 line-through opacity-60"
            }`}
          >
            <span className="w-2 h-2 rounded-full bg-purple-400 mr-1" />
            <span>3D Drift: {latestPoint.robotDrift}%</span>
          </button>
        </div>
      </div>

      {/* SVG Timeseries Chart */}
      <div className="relative bg-dark-950/80 rounded-xl p-2 border border-white/5 overflow-hidden">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-44 sm:h-52 overflow-visible select-none"
        >
          {/* Y-Axis Grid Lines & Labels */}
          {[0, 25, 50, 75, 100].map((val) => {
            const y = padTop + chartH - (val / 100) * chartH;
            return (
              <g key={`y-${val}`}>
                <line
                  x1={padLeft}
                  y1={y}
                  x2={width - padRight}
                  y2={y}
                  stroke={val === 50 ? "rgba(255,255,255,0.12)" : "rgba(255,255,255,0.05)"}
                  strokeDasharray={val === 50 ? "4 4" : "2 2"}
                />
                <text
                  x={padLeft - 8}
                  y={y + 3}
                  textAnchor="end"
                  className="fill-gray-500 font-mono text-[9px]"
                >
                  {val}%
                </text>
              </g>
            );
          })}

          {/* Calibrated Trust Safe Corridor (35% to 75%) */}
          <rect
            x={padLeft}
            y={padTop + chartH - (75 / 100) * chartH}
            width={chartW}
            height={(40 / 100) * chartH}
            fill="rgba(45,212,191,0.04)"
            stroke="rgba(45,212,191,0.15)"
            strokeDasharray="3 3"
          />
          <text
            x={width - padRight - 6}
            y={padTop + chartH - (75 / 100) * chartH + 12}
            textAnchor="end"
            className="fill-teal-400/40 font-mono text-[8px] font-semibold tracking-wider"
          >
            CALIBRATED ZONE (35-75%)
          </text>

          {/* X-Axis Time Markers (-15s, -10s, -5s, Now) */}
          {[
            { label: "-15s", frac: 0.0 },
            { label: "-10s", frac: 0.333 },
            { label: "-5s", frac: 0.666 },
            { label: "Now (0s)", frac: 1.0 },
          ].map((t) => {
            const x = padLeft + t.frac * chartW;
            return (
              <g key={`x-${t.label}`}>
                <line
                  x1={x}
                  y1={padTop}
                  x2={x}
                  y2={padTop + chartH}
                  stroke="rgba(255,255,255,0.05)"
                  strokeDasharray="2 2"
                />
                <text
                  x={x}
                  y={height - 6}
                  textAnchor={t.frac === 1.0 ? "end" : (t.frac === 0.0 ? "start" : "middle")}
                  className="fill-gray-500 font-mono text-[9px]"
                >
                  {t.label}
                </text>
              </g>
            );
          })}

          {/* Modality Paths */}
          {/* Cobot 3D Path Drift (Purple) */}
          {visibleLines.drift && (
            <path
              d={buildSvgPath((p) => p.robotDrift)}
              fill="none"
              stroke="#a855f7"
              strokeWidth="1.6"
              strokeLinecap="round"
              className="transition-all duration-150 opacity-80"
            />
          )}

          {/* Voice Acoustic Tension (Amber) */}
          {visibleLines.tension && (
            <path
              d={buildSvgPath((p) => p.acousticTension)}
              fill="none"
              stroke="#f59e0b"
              strokeWidth="1.6"
              strokeLinecap="round"
              className="transition-all duration-150 opacity-85"
            />
          )}

          {/* AU12 Smile (Emerald) */}
          {visibleLines.au12 && (
            <path
              d={buildSvgPath((p) => p.au12Smile)}
              fill="none"
              stroke="#10b981"
              strokeWidth="2.0"
              strokeLinecap="round"
              className="transition-all duration-150"
            />
          )}

          {/* AU04 Brow Furrow (Rose) */}
          {visibleLines.au04 && (
            <path
              d={buildSvgPath((p) => p.au04BrowFurrow)}
              fill="none"
              stroke="#f43f5e"
              strokeWidth="2.2"
              strokeLinecap="round"
              className="transition-all duration-150 shadow-[0_0_8px_#f43f5e]"
            />
          )}

          {/* Continuous Trust Score (Glowing Cyan - Thickest) */}
          {visibleLines.trust && (
            <g>
              <path
                d={buildSvgPath((p) => p.trustScore)}
                fill="none"
                stroke="rgba(45,212,191,0.25)"
                strokeWidth="6"
                strokeLinecap="round"
              />
              <path
                d={buildSvgPath((p) => p.trustScore)}
                fill="none"
                stroke="#2dd4bf"
                strokeWidth="2.8"
                strokeLinecap="round"
                className="transition-all duration-150 shadow-[0_0_12px_#2dd4bf]"
              />
            </g>
          )}

          {/* Pulse Node at the 'Now' Cursor */}
          {history.length > 0 && visibleLines.trust && (
            <circle
              cx={padLeft + chartW}
              cy={padTop + chartH - (latestPoint.trustScore / 100) * chartH}
              r="4"
              fill="#2dd4bf"
              className="animate-ping"
            />
          )}
        </svg>
      </div>

      {/* Rolling Stats Strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 font-mono text-[11px]">
        <div className="bg-dark-900/80 p-2 rounded-lg border border-white/5 flex items-center justify-between">
          <span className="text-gray-400">15s Trust Shift:</span>
          <span
            className={`font-bold ${
              trustDelta > 0
                ? "text-emerald-400"
                : trustDelta < 0
                ? "text-rose-400"
                : "text-gray-300"
            }`}
          >
            {trustDelta > 0 ? `+${trustDelta}%` : `${trustDelta}%`}
          </span>
        </div>

        <div className="bg-dark-900/80 p-2 rounded-lg border border-white/5 flex items-center justify-between">
          <span className="text-gray-400">Peak AU04 Brow:</span>
          <span className="font-bold text-rose-400">
            {Math.max(...history.map((p) => p.au04BrowFurrow), latestPoint.au04BrowFurrow)}%
          </span>
        </div>

        <div className="bg-dark-900/80 p-2 rounded-lg border border-white/5 flex items-center justify-between">
          <span className="text-gray-400">Peak AU12 Smile:</span>
          <span className="font-bold text-emerald-400">
            {Math.max(...history.map((p) => p.au12Smile), latestPoint.au12Smile)}%
          </span>
        </div>

        <div className="bg-dark-900/80 p-2 rounded-lg border border-white/5 flex items-center justify-between">
          <span className="text-gray-400">Max 3D Drift:</span>
          <span className="font-bold text-purple-400">
            {((Math.max(...history.map((p) => p.robotDrift), latestPoint.robotDrift) / 100) * 25).toFixed(1)} mm
          </span>
        </div>
      </div>
    </div>
  );
}
