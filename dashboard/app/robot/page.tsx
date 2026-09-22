"use client";

import React, { useState } from "react";
import Cobot3DView from "@/components/Cobot3DView";
import FaceExpressionCapture from "@/components/FaceExpressionCapture";
import { RobotState, FacialTelemetry } from "@/types/telemetry";
import {
  Bot,
  CheckCircle2,
  AlertTriangle,
  Play,
  RotateCcw,
  Activity,
  Camera,
  Shield,
  Zap,
} from "lucide-react";

export default function RobotDigitalTwinPage() {
  const [robotState, setRobotState] = useState<RobotState>({
    mode: "correct",
    error_type: "nominal",
    speed_mps: 0.8,
    drift_m: 0.02,
    torque_anomaly: 0.04,
  });

  const [facialData, setFacialData] = useState<FacialTelemetry>({
    au04_brow_furrow: 0.12,
    blink_rate_bpm: 18,
    au12_smile: 0.08,
    mouth_open: 0.05,
    valence_entropy: 0.14,
  });

  const [activeStep, setActiveStep] = useState<number>(0);
  const steps = [
    "1: Pick Hex Bolt from Tray",
    "2: Approach Workpiece Joint",
    "3: Align Fastener Axis",
    "4: Torque Tightening (12 Nm)",
    "5: Inspect Thread Seating",
    "6: Safe Retract & Handover",
  ];

  const nextStep = () => {
    setActiveStep((prev) => (prev + 1) % steps.length);
  };

  const resetCycle = () => {
    setActiveStep(0);
  };

  const isOperatorStressed = facialData.au04_brow_furrow > 0.35;

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="border-b border-white/5 pb-4">
        <h1 className="text-xl sm:text-2xl font-black text-white tracking-tight font-display flex items-center">
          <Bot className="w-6 h-6 mr-2.5 text-blue-400" />
          UR5 Cobot Digital Twin &amp; Closed-Loop Control
        </h1>
        <p className="text-xs sm:text-sm text-gray-400 mt-1">
          Interactive 3D UR5 manipulator with real-time kinematic simulation, fault injection options, and synchronized operator webcam monitoring.
        </p>
      </div>

      {/* Assembly Workflow Stepper Ribbon */}
      <div className="glass-card p-4 flex flex-wrap items-center justify-between gap-3 border-blue-500/20">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-xl bg-blue-500/10 border border-blue-500/20 text-blue-400">
            <Activity className="w-4 h-4" />
          </div>
          <div>
            <span className="text-xs font-bold text-white font-display block">
              Current Assembly Task: {steps[activeStep]}
            </span>
            <span className="text-[10px] text-gray-400 font-mono">
              Phase {activeStep + 1} of {steps.length}
            </span>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={nextStep}
            className="px-3 py-1.5 text-xs font-semibold rounded-xl bg-dark-800 hover:bg-dark-750 text-gray-200 border border-white/10 transition flex items-center space-x-1"
          >
            <Play className="w-3.5 h-3.5 text-teal-400" />
            <span>Next Phase</span>
          </button>
          <button
            onClick={resetCycle}
            className="px-3 py-1.5 text-xs font-semibold rounded-xl bg-dark-800 hover:bg-dark-750 text-gray-200 border border-white/10 transition flex items-center space-x-1"
          >
            <RotateCcw className="w-3.5 h-3.5 text-gray-400" />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* Main Grid: 3D Robot on Left (Cols 1-7), Operator Webcam on Right (Cols 8-12) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* 3D UR5 Cobot Digital Twin */}
        <div className="lg:col-span-7">
          <Cobot3DView
            onRobotStateChange={(st) => setRobotState(st)}
            trustState={
              robotState.mode === "wrong"
                ? "UNDER_TRUST"
                : isOperatorStressed
                ? "UNDER_TRUST"
                : "CALIBRATED_TRUST"
            }
          />
        </div>

        {/* Operator Biometric Camera Station */}
        <div className="lg:col-span-5 space-y-4">
          <FaceExpressionCapture
            compact
            autoStart
            title="Operator Face & Affect Cam"
            onTelemetryChange={(data) => setFacialData(data)}
          />

          {/* Real-Time Cobot Closed-Loop Coupling Card */}
          <div className="glass-card p-4 border-cyan-500/20 space-y-2.5 font-mono text-xs">
            <div className="flex items-center justify-between border-b border-white/5 pb-2">
              <span className="text-gray-300 font-bold flex items-center">
                <Shield className="w-4 h-4 mr-1.5 text-teal-400" />
                Human-Cobot Coupling
              </span>
              <span
                className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                  isOperatorStressed
                    ? "bg-rose-500/20 text-rose-300 border border-rose-500/30"
                    : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                }`}
              >
                {isOperatorStressed ? "STRESS DETECTED" : "NOMINAL HARMONY"}
              </span>
            </div>

            <div className="space-y-1.5 text-[11px]">
              <div className="flex justify-between">
                <span className="text-gray-400">Robot Mode:</span>
                <span className={robotState.mode === "wrong" ? "text-rose-400 font-bold" : "text-emerald-400 font-bold"}>
                  {robotState.mode.toUpperCase()} ({robotState.error_type})
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">AU04 Brow Stress:</span>
                <span className={isOperatorStressed ? "text-rose-400 font-bold" : "text-teal-300"}>
                  {(facialData.au04_brow_furrow * 100).toFixed(0)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Velocity Regulation:</span>
                <span className="text-white font-bold">
                  {robotState.mode === "wrong" || isOperatorStressed ? "Throttled to 0.2x - 0.5x" : "Full 1.0x (0.8 m/s)"}
                </span>
              </div>
            </div>
          </div>
        </div>

      </div>

      {/* Explanation of Correct vs Wrong Behavior */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="glass-card p-4 space-y-2 border-emerald-500/30">
          <h4 className="text-xs font-bold text-emerald-300 font-display flex items-center">
            <CheckCircle2 className="w-4 h-4 mr-1.5 text-emerald-400" />
            Correct / Nominal Operation Mode
          </h4>
          <p className="text-xs text-gray-300 leading-relaxed font-sans">
            Under correct mode, the UR5 cobot tracks the programmed minimum-jerk trajectory with tracking error &lt; 0.03m. End-effector parallel jaws securely hold the workpiece. Safety laser curtain remains in glowing green calibrated state at full collaborative speed (1.0x).
          </p>
        </div>

        <div className="glass-card p-4 space-y-2 border-rose-500/30">
          <h4 className="text-xs font-bold text-rose-300 font-display flex items-center">
            <AlertTriangle className="w-4 h-4 mr-1.5 text-rose-400" />
            Wrong / Fault Injected Mode
          </h4>
          <p className="text-xs text-gray-300 leading-relaxed font-sans">
            Injecting errors simulates industrial anomalies: gripper slip (dropped workpiece), path drift (0.35m trajectory swerve), overspeed (1.45 m/s), or joint torque spikes. The closed-loop mitigation engine automatically clamps velocity and engages safety standoffs.
          </p>
        </div>
      </div>
    </div>
  );
}
