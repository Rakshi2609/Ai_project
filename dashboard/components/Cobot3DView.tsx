"use client";

import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { Bot, CheckCircle2, AlertTriangle, ShieldCheck, RefreshCw, Eye, FastForward } from "lucide-react";
import { RobotState } from "@/types/telemetry";

interface Cobot3DViewProps {
  onRobotStateChange?: (state: RobotState) => void;
  trustState?: string;
}

export default function Cobot3DView({
  onRobotStateChange,
  trustState = "CALIBRATED_TRUST",
}: Cobot3DViewProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);

  // Mesh refs for animated joint kinematics
  const upperArmRef = useRef<THREE.Mesh | null>(null);
  const forearmRef = useRef<THREE.Mesh | null>(null);
  const gripperLeftRef = useRef<THREE.Mesh | null>(null);
  const gripperRightRef = useRef<THREE.Mesh | null>(null);
  const workpieceRef = useRef<THREE.Mesh | null>(null);
  const laserCurtainRef = useRef<THREE.Mesh | null>(null);
  const photonRef = useRef<THREE.Mesh | null>(null);
  const animIdRef = useRef<number | null>(null);

  // Robot State: Correct vs Wrong Options
  const [robotMode, setRobotMode] = useState<"correct" | "wrong">("correct");
  const [faultType, setFaultType] = useState<
    "nominal" | "gripper_slip" | "trajectory_drift" | "excessive_speed" | "torque_anomaly"
  >("nominal");

  const [metrics, setMetrics] = useState({
    speed: 0.8,
    drift: 0.02,
    torque: 0.04,
  });

  // Handle Mode Change (Correct vs Wrong)
  const setCorrectMode = () => {
    setRobotMode("correct");
    setFaultType("nominal");
    const newState: RobotState = {
      mode: "correct",
      error_type: "nominal",
      speed_mps: 0.8,
      drift_m: 0.02,
      torque_anomaly: 0.04,
    };
    setMetrics({ speed: 0.8, drift: 0.02, torque: 0.04 });
    if (onRobotStateChange) onRobotStateChange(newState);
  };

  const setWrongMode = (
    type: "gripper_slip" | "trajectory_drift" | "excessive_speed" | "torque_anomaly"
  ) => {
    setRobotMode("wrong");
    setFaultType(type);

    let speed = 0.8;
    let drift = 0.02;
    let torque = 0.04;

    if (type === "gripper_slip") {
      drift = 0.12;
      speed = 0.2;
      torque = 0.18;
    } else if (type === "trajectory_drift") {
      drift = 0.35;
      speed = 0.75;
      torque = 0.22;
    } else if (type === "excessive_speed") {
      speed = 1.45;
      drift = 0.08;
      torque = 0.45;
    } else if (type === "torque_anomaly") {
      speed = 0.5;
      drift = 0.05;
      torque = 0.88;
    }

    setMetrics({ speed, drift, torque });
    const newState: RobotState = {
      mode: "wrong",
      error_type: type,
      speed_mps: speed,
      drift_m: drift,
      torque_anomaly: torque,
    };
    if (onRobotStateChange) onRobotStateChange(newState);
  };

  // Trigger Active Trust Recovery
  const triggerActiveRecovery = () => {
    setCorrectMode();
  };

  // Initialize Three.js UR5 Scene
  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth || 600;
    const height = container.clientHeight || 400;

    const scene = new THREE.Scene();
    sceneRef.current = scene;
    scene.background = new THREE.Color(0x060913);
    scene.fog = new THREE.FogExp2(0x060913, 0.18);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 50);
    camera.position.set(2.4, 2.0, 3.2);
    camera.lookAt(0.1, 0.4, 0.0);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    rendererRef.current = renderer;

    container.replaceChildren(renderer.domElement);

    // Grid Floor
    const grid = new THREE.GridHelper(8, 24, 0x2dd4bf, 0x1e293b);
    grid.position.y = -0.5;
    scene.add(grid);

    // Industrial Lighting
    const ambLight = new THREE.AmbientLight(0xffffff, 0.85);
    scene.add(ambLight);

    const dirLight = new THREE.DirectionalLight(0x2dd4bf, 1.4);
    dirLight.position.set(4, 6, 4);
    scene.add(dirLight);

    const blueLight = new THREE.DirectionalLight(0x3b82f6, 1.0);
    blueLight.position.set(-4, 3, -3);
    scene.add(blueLight);

    // Materials
    const cobotGreyMat = new THREE.MeshStandardMaterial({ color: 0x475569, roughness: 0.3, metalness: 0.7 });
    const cobotCyanMat = new THREE.MeshStandardMaterial({ color: 0x0ea5e9, roughness: 0.2, metalness: 0.8 });
    const darkJointMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.5, metalness: 0.9 });
    const workpieceMat = new THREE.MeshStandardMaterial({ color: 0xf59e0b, roughness: 0.2, metalness: 0.5 });

    // Table
    const tableGeo = new THREE.BoxGeometry(2.4, 0.1, 1.4);
    const tableMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.4 });
    const table = new THREE.Mesh(tableGeo, tableMat);
    table.position.set(0, -0.45, 0);
    scene.add(table);

    // UR5 Base Pedestal
    const baseGeo = new THREE.CylinderGeometry(0.16, 0.2, 0.15, 32);
    const ur5Base = new THREE.Mesh(baseGeo, darkJointMat);
    ur5Base.position.set(-0.35, -0.32, 0);
    scene.add(ur5Base);

    // Shoulder
    const shoulderGeo = new THREE.CylinderGeometry(0.13, 0.13, 0.22, 24);
    shoulderGeo.rotateZ(Math.PI / 2);
    const shoulderJoint = new THREE.Mesh(shoulderGeo, cobotCyanMat);
    shoulderJoint.position.set(-0.35, -0.18, 0);
    scene.add(shoulderJoint);

    // Upper Arm
    const upperGeo = new THREE.CylinderGeometry(0.08, 0.08, 0.65, 24);
    const upperArm = new THREE.Mesh(upperGeo, cobotGreyMat);
    upperArm.position.set(-0.15, 0.12, 0);
    upperArm.rotation.z = -Math.PI / 4;
    scene.add(upperArm);
    upperArmRef.current = upperArm;

    // Elbow Joint
    const elbowGeo = new THREE.CylinderGeometry(0.11, 0.11, 0.2, 24);
    elbowGeo.rotateZ(Math.PI / 2);
    const elbowJoint = new THREE.Mesh(elbowGeo, cobotCyanMat);
    elbowJoint.position.set(0.08, 0.35, 0);
    scene.add(elbowJoint);

    // Forearm
    const foreGeo = new THREE.CylinderGeometry(0.065, 0.065, 0.55, 24);
    const forearm = new THREE.Mesh(foreGeo, cobotGreyMat);
    forearm.position.set(0.28, 0.25, 0);
    forearm.rotation.z = Math.PI / 2.8;
    scene.add(forearm);
    forearmRef.current = forearm;

    // Wrist
    const wristGeo = new THREE.CylinderGeometry(0.06, 0.06, 0.12, 20);
    const wrist = new THREE.Mesh(wristGeo, darkJointMat);
    wrist.position.set(0.48, 0.12, 0);
    scene.add(wrist);

    // Gripper Base & Parallel Fingers
    const gripBaseGeo = new THREE.BoxGeometry(0.08, 0.04, 0.12);
    const gripBase = new THREE.Mesh(gripBaseGeo, darkJointMat);
    gripBase.position.set(0.53, 0.12, 0);
    scene.add(gripBase);

    const fingerGeo = new THREE.BoxGeometry(0.06, 0.1, 0.015);
    const gripperLeft = new THREE.Mesh(fingerGeo, cobotCyanMat);
    gripperLeft.position.set(0.56, 0.08, -0.04);
    scene.add(gripperLeft);
    gripperLeftRef.current = gripperLeft;

    const gripperRight = new THREE.Mesh(fingerGeo, cobotCyanMat);
    gripperRight.position.set(0.56, 0.08, 0.04);
    scene.add(gripperRight);
    gripperRightRef.current = gripperRight;

    // Workpiece
    const workGeo = new THREE.CylinderGeometry(0.035, 0.035, 0.08, 16);
    const workpiece = new THREE.Mesh(workGeo, workpieceMat);
    workpiece.position.set(0.56, 0.07, 0);
    scene.add(workpiece);
    workpieceRef.current = workpiece;

    // Safety Laser Curtain (Cylindrical Bounding Volume)
    const curtainGeo = new THREE.CylinderGeometry(0.9, 0.9, 1.2, 32, 1, true);
    const curtainMat = new THREE.MeshBasicMaterial({
      color: 0x10b981,
      transparent: true,
      opacity: 0.25,
      side: THREE.DoubleSide,
      wireframe: false,
    });
    const laserCurtain = new THREE.Mesh(curtainGeo, curtainMat);
    laserCurtain.position.set(0.1, 0.15, 0);
    scene.add(laserCurtain);
    laserCurtainRef.current = laserCurtain;

    // Photon Particle on Planned Path
    const photonGeo = new THREE.SphereGeometry(0.025, 16, 16);
    const photonMat = new THREE.MeshBasicMaterial({ color: 0x00f2fe });
    const photon = new THREE.Mesh(photonGeo, photonMat);
    scene.add(photon);
    photonRef.current = photon;

    // Animation Loop
    let photonProg = 0;
    const animate = () => {
      animIdRef.current = requestAnimationFrame(animate);

      // Pulse laser curtain
      if (laserCurtainRef.current) {
        const time = Date.now() * 0.002;
        (laserCurtainRef.current.material as THREE.MeshBasicMaterial).opacity = 0.2 + Math.sin(time) * 0.08;
      }

      // Animate photon along path
      photonProg = (photonProg + 0.008) % 1.0;
      if (photonRef.current) {
        const x = -0.3 + photonProg * 0.8;
        const y = 0.2 + Math.sin(photonProg * Math.PI) * 0.2;
        const z = photonProg * 0.1;
        photonRef.current.position.set(x, y, z);
      }

      renderer.render(scene, camera);
    };
    animate();

    // Resize handler
    const handleResize = () => {
      if (!container || !renderer || !camera) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      if (animIdRef.current) cancelAnimationFrame(animIdRef.current);
      renderer.dispose();
    };
  }, []);

  // Update Three.js mesh positions and colors when robot state changes
  useEffect(() => {
    if (!laserCurtainRef.current) return;
    const curtainMat = laserCurtainRef.current.material as THREE.MeshBasicMaterial;

    if (robotMode === "correct") {
      curtainMat.color.setHex(0x10b981); // Emerald
      if (gripperLeftRef.current && gripperRightRef.current && workpieceRef.current) {
        gripperLeftRef.current.position.z = -0.035;
        gripperRightRef.current.position.z = 0.035;
        workpieceRef.current.position.set(0.56, 0.07, 0);
      }
    } else {
      // Wrong / Fault injected
      if (faultType === "gripper_slip") {
        curtainMat.color.setHex(0xf43f5e); // Rose alert
        if (gripperLeftRef.current && gripperRightRef.current && workpieceRef.current) {
          gripperLeftRef.current.position.z = -0.07;
          gripperRightRef.current.position.z = 0.07;
          workpieceRef.current.position.set(0.56, -0.38, 0); // Dropped onto table
        }
      } else if (faultType === "trajectory_drift") {
        curtainMat.color.setHex(0xf59e0b); // Amber
        if (forearmRef.current) forearmRef.current.rotation.z = Math.PI / 2.2;
      } else if (faultType === "excessive_speed") {
        curtainMat.color.setHex(0xc084fc); // Purple
      } else if (faultType === "torque_anomaly") {
        curtainMat.color.setHex(0xf43f5e); // Rose
      }
    }
  }, [robotMode, faultType]);

  return (
    <div className="glass-card p-4 space-y-4 border-blue-500/30 shadow-2xl relative">
      {/* Header and Mode Indicator */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/5 pb-3">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-400">
            <Bot className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-xs font-bold text-white font-display tracking-wide flex items-center">
              Universal Robots UR5 Digital Twin
              <span className="ml-2 px-2 py-0.5 rounded text-[10px] font-mono bg-blue-500/20 text-blue-300 border border-blue-500/30">
                Kinematics &amp; Closed-Loop Control
              </span>
            </h3>
            <p className="text-[10px] text-gray-400 font-mono">Real-time Joint Angle &amp; End-Effector Telemetry</p>
          </div>
        </div>

        {/* Big Correct vs Wrong Toggle Switch */}
        <div className="flex items-center space-x-1.5 bg-dark-900 p-1 rounded-xl border border-white/10">
          <button
            onClick={setCorrectMode}
            className={`px-3 py-1.5 text-xs font-bold rounded-lg transition flex items-center space-x-1.5 font-display ${
              robotMode === "correct"
                ? "bg-gradient-to-r from-emerald-500 to-teal-400 text-dark-950 shadow-[0_0_12px_rgba(16,185,129,0.5)]"
                : "text-gray-400 hover:text-white"
            }`}
          >
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>CORRECT (Nominal)</span>
          </button>

          <button
            onClick={() => setWrongMode("gripper_slip")}
            className={`px-3 py-1.5 text-xs font-bold rounded-lg transition flex items-center space-x-1.5 font-display ${
              robotMode === "wrong"
                ? "bg-gradient-to-r from-rose-500 to-amber-500 text-white shadow-[0_0_12px_rgba(244,63,94,0.5)]"
                : "text-gray-400 hover:text-white"
            }`}
          >
            <AlertTriangle className="w-3.5 h-3.5" />
            <span>WRONG (Inject Fault)</span>
          </button>
        </div>
      </div>

      {/* 3D WebGL Viewport */}
      <div className="relative h-72 sm:h-80 w-full rounded-2xl overflow-hidden border border-white/10 bg-dark-950 shadow-inner">
        <div ref={mountRef} className="w-full h-full" />

        {/* Floating Telemetry Badge */}
        <div className="absolute top-3 left-3 bg-dark-950/85 backdrop-blur-md px-3 py-2 rounded-xl border border-white/10 text-[11px] font-mono space-y-1 pointer-events-none">
          <div className="flex justify-between items-center space-x-4">
            <span className="text-gray-400">Robot Status:</span>
            <span className={`font-bold ${robotMode === "correct" ? "text-emerald-400" : "text-rose-400"}`}>
              {robotMode === "correct" ? "CORRECT / NOMINAL" : `FAULT: ${faultType.toUpperCase()}`}
            </span>
          </div>
          <div className="flex justify-between items-center space-x-4">
            <span className="text-gray-400">End-Effector Drift:</span>
            <span className="text-white font-bold">{metrics.drift.toFixed(2)} m</span>
          </div>
          <div className="flex justify-between items-center space-x-4">
            <span className="text-gray-400">Velocity:</span>
            <span className="text-cyan-300 font-bold">{metrics.speed.toFixed(2)} m/s</span>
          </div>
        </div>

        {/* Fault Selector Sub-Toolbar (When WRONG is active) */}
        {robotMode === "wrong" && (
          <div className="absolute bottom-3 left-3 right-3 bg-dark-950/90 backdrop-blur-md p-2 rounded-xl border border-rose-500/40 flex flex-wrap items-center justify-between gap-2">
            <span className="text-[11px] font-mono text-rose-300 font-bold flex items-center">
              <AlertTriangle className="w-3.5 h-3.5 mr-1.5 text-rose-400" />
              SELECT FAULT TYPE:
            </span>
            <div className="flex items-center space-x-1.5">
              <button
                onClick={() => setWrongMode("gripper_slip")}
                className={`px-2.5 py-1 text-[11px] font-mono rounded-lg transition ${
                  faultType === "gripper_slip"
                    ? "bg-rose-500 text-white font-bold shadow-[0_0_8px_#f43f5e]"
                    : "bg-dark-800 text-gray-300 hover:bg-dark-750"
                }`}
              >
                1: Gripper Slip
              </button>
              <button
                onClick={() => setWrongMode("trajectory_drift")}
                className={`px-2.5 py-1 text-[11px] font-mono rounded-lg transition ${
                  faultType === "trajectory_drift"
                    ? "bg-amber-500 text-dark-950 font-bold shadow-[0_0_8px_#fbbf24]"
                    : "bg-dark-800 text-gray-300 hover:bg-dark-750"
                }`}
              >
                2: Trajectory Drift
              </button>
              <button
                onClick={() => setWrongMode("excessive_speed")}
                className={`px-2.5 py-1 text-[11px] font-mono rounded-lg transition ${
                  faultType === "excessive_speed"
                    ? "bg-purple-500 text-white font-bold shadow-[0_0_8px_#c084fc]"
                    : "bg-dark-800 text-gray-300 hover:bg-dark-750"
                }`}
              >
                3: Overspeed
              </button>
              <button
                onClick={() => setWrongMode("torque_anomaly")}
                className={`px-2.5 py-1 text-[11px] font-mono rounded-lg transition ${
                  faultType === "torque_anomaly"
                    ? "bg-rose-600 text-white font-bold shadow-[0_0_8px_#e11d48]"
                    : "bg-dark-800 text-gray-300 hover:bg-dark-750"
                }`}
              >
                4: Torque Spike
              </button>
            </div>
            <button
              onClick={triggerActiveRecovery}
              className="px-3 py-1 text-[11px] font-bold rounded-lg bg-emerald-500 hover:bg-emerald-400 text-dark-950 flex items-center space-x-1 transition shadow-sm font-display"
            >
              <ShieldCheck className="w-3.5 h-3.5" />
              <span>Recover Nominal</span>
            </button>
          </div>
        )}
      </div>

      {/* Kinematics Quick Telemetry Bar */}
      <div className="grid grid-cols-3 gap-2 text-xs font-mono">
        <div className="bg-dark-900/90 p-2.5 rounded-xl border border-white/5">
          <span className="text-[10px] text-gray-400 block">Linear Speed</span>
          <span className="text-white font-bold">{metrics.speed.toFixed(2)} m/s</span>
        </div>
        <div className="bg-dark-900/90 p-2.5 rounded-xl border border-white/5">
          <span className="text-[10px] text-gray-400 block">Trajectory Drift</span>
          <span className={`font-bold ${metrics.drift > 0.1 ? "text-rose-400" : "text-emerald-400"}`}>
            {metrics.drift.toFixed(2)} m
          </span>
        </div>
        <div className="bg-dark-900/90 p-2.5 rounded-xl border border-white/5">
          <span className="text-[10px] text-gray-400 block">Joint Torque</span>
          <span className={`font-bold ${metrics.torque > 0.5 ? "text-rose-400" : "text-cyan-300"}`}>
            {(metrics.torque * 100).toFixed(0)}% Nominal
          </span>
        </div>
      </div>
    </div>
  );
}
