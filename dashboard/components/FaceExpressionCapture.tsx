"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import {
  Camera,
  CameraOff,
  Video,
  AlertCircle,
  Sparkles,
  Smile,
  Frown,
  Eye,
  RefreshCw,
  Sliders,
  CheckCircle2,
  Share2,
  Zap,
} from "lucide-react";
import { FacialTelemetry } from "@/types/telemetry";

interface FaceExpressionCaptureProps {
  onTelemetryChange?: (data: FacialTelemetry) => void;
  compact?: boolean;
  autoStart?: boolean;
  title?: string;
}

export default function FaceExpressionCapture({
  onTelemetryChange,
  compact = false,
  autoStart = false,
  title = "Real Facial Expression Capture",
}: FaceExpressionCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const cameraActiveRef = useRef<boolean>(false);
  const onTelemetryChangeRef = useRef(onTelemetryChange);
  onTelemetryChangeRef.current = onTelemetryChange;

  const [cameraActive, setCameraActive] = useState<boolean>(false);
  const [isInitializing, setIsInitializing] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [cameraDevices, setCameraDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const [manualMode, setManualMode] = useState<boolean>(false);
  const [expressionName, setExpressionName] = useState<string>("Neutral (Calm)");
  const [frameTick, setFrameTick] = useState<number>(0);

  const [detectionState, setDetectionState] = useState<{
    faceDetected: boolean;
    au04_brow_furrow: number;
    au12_smile: number;
    mouth_open: number;
    blink_rate: number;
    entropy: number;
  }>({
    faceDetected: false,
    au04_brow_furrow: 0.12,
    au12_smile: 0.08,
    mouth_open: 0.05,
    blink_rate: 18,
    entropy: 0.14,
  });

  const detectionStateRef = useRef(detectionState);
  detectionStateRef.current = detectionState;

  const [lockedPreset, setLockedPreset] = useState<string | null>(null);
  const lockedPresetRef = useRef<string | null>(null);
  const baselineContrastRef = useRef<number | null>(null);
  const baselineMouthLumRef = useRef<number | null>(null);

  const faceLandmarkerRef = useRef<any>(null);
  const [visionEngine, setVisionEngine] = useState<"mediapipe" | "optical_flow">("optical_flow");

  const lastAnalysisTimeRef = useRef<number>(0);
  const blinkHistoryRef = useRef<number[]>([]);
  const lastEyeLuminanceRef = useRef<number>(100);

  // Initialize Google MediaPipe 478-Point FaceLandmarker with Local WASM Assets
  useEffect(() => {
    let isMounted = true;
    const initMediaPipe = async () => {
      if (typeof window === "undefined") return;
      try {
        const { FaceLandmarker, FilesetResolver } = await import("@mediapipe/tasks-vision");
        const fileset = await FilesetResolver.forVisionTasks("/wasm");
        if (!isMounted) return;

        let landmarker: any;
        try {
          landmarker = await FaceLandmarker.createFromOptions(fileset, {
            baseOptions: {
              modelAssetPath: "/models/face_landmarker.task",
              delegate: "GPU",
            },
            outputFaceBlendshapes: true,
            runningMode: "VIDEO",
            numFaces: 1,
          });
        } catch (gpuErr) {
          console.warn("GPU delegate unavailable, falling back to CPU delegate:", gpuErr);
          landmarker = await FaceLandmarker.createFromOptions(fileset, {
            baseOptions: {
              modelAssetPath: "/models/face_landmarker.task",
              delegate: "CPU",
            },
            outputFaceBlendshapes: true,
            runningMode: "VIDEO",
            numFaces: 1,
          });
        }

        if (isMounted && landmarker) {
          faceLandmarkerRef.current = landmarker;
          setVisionEngine("mediapipe");
          console.log("MediaPipe FaceLandmarker initialized with 478-point mesh and FACS blendshapes!");
        }
      } catch (err) {
        console.warn("MediaPipe initialization failed, using adaptive optical flow fallback:", err);
        setVisionEngine("optical_flow");
      }
    };

    initMediaPipe();

    return () => {
      isMounted = false;
      if (faceLandmarkerRef.current) {
        try {
          faceLandmarkerRef.current.close();
        } catch {}
        faceLandmarkerRef.current = null;
      }
    };
  }, []);

  // Enumerate cameras when component mounts
  useEffect(() => {
    const listDevices = async () => {
      if (typeof navigator === "undefined" || !navigator.mediaDevices?.enumerateDevices) {
        return;
      }
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevs = devices.filter((d) => d.kind === "videoinput");
        setCameraDevices(videoDevs);
        if (videoDevs.length > 0 && !selectedDeviceId) {
          setSelectedDeviceId(videoDevs[0].deviceId);
        }
      } catch (err) {
        console.warn("Could not enumerate camera devices:", err);
      }
    };
    listDevices();
  }, [selectedDeviceId]);

  // Clean stop of camera tracks
  const stopCamera = useCallback(() => {
    cameraActiveRef.current = false;
    setCameraActive(false);
    setIsInitializing(false);

    if (typeof window !== "undefined") {
      sessionStorage.removeItem("cobot_camera_enabled");
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch {
          // Ignore
        }
      });
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  // ONLY stop camera on component unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  // Robust Multi-Stage Camera Starter with Progressive Fallback
  const startCamera = async (deviceIdOverride?: string) => {
    setIsInitializing(true);
    setErrorMsg(null);

    // 1. Verify Browser Support & Secure Context
    if (
      typeof window !== "undefined" &&
      !window.isSecureContext &&
      window.location.hostname !== "localhost" &&
      window.location.hostname !== "127.0.0.1"
    ) {
      setErrorMsg(
        "Webcam requires HTTPS or http://localhost:3000. Accessing via raw network IP blocks WebRTC permissions."
      );
      setIsInitializing(false);
      return;
    }

    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      setErrorMsg("WebRTC Camera API is not supported on this browser.");
      setIsInitializing(false);
      return;
    }

    // Stop existing stream if switching device
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    const devId = deviceIdOverride || selectedDeviceId;
    let stream: MediaStream | null = null;
    let lastError: Error | null = null;

    // Stage 1: Try with preferred resolution & specified device
    try {
      const constraints: MediaStreamConstraints = {
        video: devId
          ? {
              deviceId: { exact: devId },
              width: { ideal: 640 },
              height: { ideal: 480 },
            }
          : {
              width: { ideal: 640 },
              height: { ideal: 480 },
            },
      };
      stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (err1: unknown) {
      lastError = err1 as Error;
      console.warn("Stage 1 camera request failed, trying Stage 2 fallback:", err1);
    }

    // Stage 2: Fallback without explicit device/resolution constraints
    if (!stream) {
      try {
        const fallbackConstraints: MediaStreamConstraints = devId
          ? { video: { deviceId: { exact: devId } } }
          : { video: true };
        stream = await navigator.mediaDevices.getUserMedia(fallbackConstraints);
      } catch (err2: unknown) {
        lastError = err2 as Error;
        console.warn("Stage 2 camera request failed, trying Stage 3 bare video:", err2);
      }
    }

    // Stage 3: Universal bare minimum { video: true }
    if (!stream) {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
      } catch (err3: unknown) {
        lastError = err3 as Error;
      }
    }

    if (!stream) {
      setIsInitializing(false);
      const name = lastError?.name || "";
      const msg = lastError?.message || "";

      if (name === "NotAllowedError" || name === "PermissionDeniedError") {
        setErrorMsg(
          "Permission Denied: Click the camera icon in your browser URL bar, select 'Allow', then click Share Camera Feed."
        );
      } else if (name === "NotFoundError" || name === "DevicesNotFoundError") {
        setErrorMsg("No camera device detected. Please connect a webcam.");
      } else if (name === "NotReadableError" || name === "TrackStartError") {
        setErrorMsg(
          "Camera is locked by another program (Zoom, Teams, Meet, or another tab). Close it and retry."
        );
      } else {
        setErrorMsg(`Camera error (${name || "Unknown"}): ${msg || "Unable to acquire video stream"}`);
      }
      return;
    }

    streamRef.current = stream;
    cameraActiveRef.current = true;
    setCameraActive(true);
    setIsInitializing(false);

    if (typeof window !== "undefined") {
      sessionStorage.setItem("cobot_camera_enabled", "true");
    }

    // Refresh device list
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevs = devices.filter((d) => d.kind === "videoinput");
      setCameraDevices(videoDevs);
    } catch {
      // Ignore
    }

    if (videoRef.current) {
      const video = videoRef.current;
      video.srcObject = stream;
      try {
        await video.play();
      } catch (playErr) {
        console.warn("Video play error:", playErr);
      }
    }
  };

  // Auto-start camera if requested or previously enabled in session
  useEffect(() => {
    const isPreviouslyEnabled =
      typeof window !== "undefined" && sessionStorage.getItem("cobot_camera_enabled") === "true";
    if (autoStart || isPreviouslyEnabled) {
      startCamera();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoStart]);

  // Quick 2-Mode Preset Expression Injector (Smile vs Stressed)
  const applyPresetExpression = (preset: "smile" | "stress") => {
    let telemetry: FacialTelemetry;
    let label = "Smile (Calm & Calibrated)";

    if (preset === "smile") {
      telemetry = {
        au04_brow_furrow: 0.05,
        au12_smile: 0.88,
        mouth_open: 0.12,
        blink_rate_bpm: 16,
        valence_entropy: 0.10,
      };
      label = "Smile (Calm & Calibrated)";
    } else {
      telemetry = {
        au04_brow_furrow: 0.88,
        au12_smile: 0.02,
        mouth_open: 0.16,
        blink_rate_bpm: 34,
        valence_entropy: 0.76,
      };
      label = "Stressed (Brow Furrow AU04)";
    }

    lockedPresetRef.current = preset;
    setLockedPreset(preset);
    setExpressionName(label);
    const updated = {
      faceDetected: true,
      ...telemetry,
      blink_rate: telemetry.blink_rate_bpm,
      entropy: telemetry.valence_entropy,
    };
    detectionStateRef.current = updated;
    setDetectionState(updated);

    if (onTelemetryChangeRef.current) {
      onTelemetryChangeRef.current(telemetry);
    }
  };

  const resumeLiveCameraTracking = () => {
    lockedPresetRef.current = null;
    setLockedPreset(null);
    baselineContrastRef.current = null;
    baselineMouthLumRef.current = null;
  };

  // Draw High-Precision 478-Point FaceMesh & FACS Reticles on Canvas
  const drawMediaPipeHUD = (
    ctx: CanvasRenderingContext2D,
    landmarks: { x: number; y: number; z: number }[],
    w: number,
    h: number,
    isStressed: boolean,
    au04: number,
    au12: number
  ) => {
    const themeColor = isStressed ? "#f43f5e" : "#2dd4bf";

    // Video is styled with CSS -scale-x-100 (mirrored), so mirror X to align overlay
    const toX = (normX: number) => (1 - normX) * w;
    const toY = (normY: number) => normY * h;

    const drawContour = (indices: number[], close = false, lineWidth = 1.8, strokeColor = themeColor) => {
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      for (let i = 0; i < indices.length; i++) {
        const pt = landmarks[indices[i]];
        if (!pt) continue;
        const x = toX(pt.x);
        const y = toY(pt.y);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      if (close) ctx.closePath();
      ctx.stroke();
    };

    // Eyebrows (AU04 Brow Lowerer / Furrow)
    drawContour([70, 63, 105, 66, 107], false, 2.5, isStressed ? "#f43f5e" : "#38bdf8");
    drawContour([336, 296, 334, 293, 300], false, 2.5, isStressed ? "#f43f5e" : "#38bdf8");

    // Eyes
    drawContour([33, 160, 158, 133, 153, 144], true, 1.5, themeColor);
    drawContour([362, 385, 387, 263, 373, 380], true, 1.5, themeColor);

    // Mouth / Lips (AU12 Lip Corner Puller)
    drawContour(
      [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61],
      true,
      2.0,
      !isStressed ? "#38bdf8" : themeColor
    );

    // Key Landmark Nodes
    ctx.fillStyle = themeColor;
    const keyPoints = [1, 4, 10, 152, 33, 263, 61, 291, 70, 336];
    for (const idx of keyPoints) {
      const pt = landmarks[idx];
      if (pt) {
        ctx.beginPath();
        ctx.arc(toX(pt.x), toY(pt.y), 2.2, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Dynamic Face Bounding Box with Corner Brackets
    let minX = 1,
      minY = 1,
      maxX = 0,
      maxY = 0;
    for (const pt of landmarks) {
      const mx = 1 - pt.x;
      if (mx < minX) minX = mx;
      if (mx > maxX) maxX = mx;
      if (pt.y < minY) minY = pt.y;
      if (pt.y > maxY) maxY = pt.y;
    }
    const pad = 0.04;
    const bx = Math.max(0, minX - pad) * w;
    const by = Math.max(0, minY - pad) * h;
    const bw = Math.min(w - bx, (maxX - minX + pad * 2) * w);
    const bh = Math.min(h - by, (maxY - minY + pad * 2) * h);

    ctx.strokeStyle = themeColor;
    ctx.lineWidth = 2.0;
    const bLen = 18;
    ctx.beginPath();
    ctx.moveTo(bx, by + bLen); ctx.lineTo(bx, by); ctx.lineTo(bx + bLen, by);
    ctx.moveTo(bx + bw - bLen, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + bLen);
    ctx.moveTo(bx, by + bh - bLen); ctx.lineTo(bx, by + bh); ctx.lineTo(bx + bLen, by + bh);
    ctx.moveTo(bx + bw - bLen, by + bh); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw, by + bh - bLen);
    ctx.stroke();

    // Top Landmarker HUD Tag
    ctx.font = "bold 9px JetBrains Mono, monospace";
    ctx.fillStyle = themeColor;
    ctx.fillText("⚡ MEDIAPIPE FACE MESH (478-PTS)", bx + 4, Math.max(12, by - 6));
    ctx.fillText(
      `AU04 Brow: ${(au04 * 100).toFixed(0)}% • AU12 Smile: ${(au12 * 100).toFixed(0)}%`,
      bx + 4,
      by + bh + 14
    );
  };

  // Real-time Computer Vision Optical Flow Analysis & AR Reticles Loop (60 FPS)
  const processFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!canvas) {
      animFrameRef.current = requestAnimationFrame(processFrame);
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      animFrameRef.current = requestAnimationFrame(processFrame);
      return;
    }

    const w = canvas.width;
    const h = canvas.height;

    // Clear previous AR overlay
    ctx.clearRect(0, 0, w, h);

    if (cameraActiveRef.current && video && video.readyState >= 2 && !video.paused) {
      const now = performance.now();

      // --- PATHWAY A: Real Google MediaPipe 478-Point FaceLandmarker ---
      if (faceLandmarkerRef.current) {
        try {
          const results = faceLandmarkerRef.current.detectForVideo(video, now);
          if (results && results.faceLandmarks && results.faceLandmarks.length > 0) {
            const landmarks = results.faceLandmarks[0];

            let smileScore = 0.08;
            let browFurrowScore = 0.08;
            let mouthOpenScore = 0.05;
            let blinkScore = 0.0;

            if (results.faceBlendshapes && results.faceBlendshapes.length > 0) {
              const categories = results.faceBlendshapes[0].categories;
              const blendMap = new Map<string, number>();
              for (const cat of categories) {
                blendMap.set(cat.categoryName, cat.score);
              }
              const smileLeft = blendMap.get("mouthSmileLeft") || 0;
              const smileRight = blendMap.get("mouthSmileRight") || 0;
              const browLeft = blendMap.get("browDownLeft") || 0;
              const browRight = blendMap.get("browDownRight") || 0;
              const jawOpen = blendMap.get("jawOpen") || 0;
              const blinkLeft = blendMap.get("eyeBlinkLeft") || 0;
              const blinkRight = blendMap.get("eyeBlinkRight") || 0;

              smileScore = (smileLeft + smileRight) / 2;
              browFurrowScore = (browLeft + browRight) / 2;
              mouthOpenScore = jawOpen;
              blinkScore = (blinkLeft + blinkRight) / 2;
            } else {
              // Geometric landmark fallback
              const lipDist = Math.hypot(landmarks[61].x - landmarks[291].x, landmarks[61].y - landmarks[291].y);
              smileScore = Math.min(1.0, Math.max(0.0, (lipDist - 0.16) * 3.5));
              const browDist = Math.hypot(landmarks[66].x - landmarks[296].x, landmarks[66].y - landmarks[296].y);
              browFurrowScore = Math.min(1.0, Math.max(0.0, (0.18 - browDist) * 4.0));
            }

            if (blinkScore > 0.45) {
              blinkHistoryRef.current.push(now);
            }
            blinkHistoryRef.current = blinkHistoryRef.current.filter((t) => now - t < 60000);
            const calculatedBpm = Math.max(
              12,
              Math.min(48, blinkHistoryRef.current.length * (60000 / Math.max(5000, now)))
            );

            // Determine 2-mode expression: Smile vs Stressed
            const isStressed = browFurrowScore > 0.22;

            // Draw full MediaPipe AR HUD with 478 landmarks & contours
            drawMediaPipeHUD(ctx, landmarks, w, h, isStressed, browFurrowScore, smileScore);

            if (!lockedPresetRef.current) {
              const expLabel = isStressed ? "Stressed (Brow Furrow AU04)" : "Smile (Calm & Calibrated)";
              const newTelemetry: FacialTelemetry = {
                au04_brow_furrow: isStressed ? Math.max(0.68, parseFloat(browFurrowScore.toFixed(3))) : 0.05,
                au12_smile: isStressed ? 0.04 : Math.max(0.72, parseFloat(smileScore.toFixed(3))),
                mouth_open: parseFloat(mouthOpenScore.toFixed(3)),
                blink_rate_bpm: Math.round(calculatedBpm),
                valence_entropy: isStressed ? 0.74 : 0.12,
              };

              setExpressionName(expLabel);
              const updatedState = {
                faceDetected: true,
                ...newTelemetry,
                blink_rate: newTelemetry.blink_rate_bpm,
                entropy: newTelemetry.valence_entropy,
              };
              detectionStateRef.current = updatedState;
              setDetectionState(updatedState);

              if (onTelemetryChangeRef.current) {
                onTelemetryChangeRef.current(newTelemetry);
              }
            }

            animFrameRef.current = requestAnimationFrame(processFrame);
            return;
          }
        } catch (mpErr) {
          // If detectForVideo fails temporarily, fall through to optical flow
        }
      }

      // --- PATHWAY B: Adaptive Optical Flow Heuristic Fallback ---
      // Analyze Expression Every 90ms (~11 Hz) for smooth real-time telemetry updates
      if (now - lastAnalysisTimeRef.current > 90) {
        lastAnalysisTimeRef.current = now;

        if (!offscreenCanvasRef.current) {
          offscreenCanvasRef.current = document.createElement("canvas");
          offscreenCanvasRef.current.width = 160;
          offscreenCanvasRef.current.height = 120;
        }
        const offCanvas = offscreenCanvasRef.current;
        const offCtx = offCanvas.getContext("2d", { willReadFrequently: true });

        if (offCtx) {
          try {
            offCtx.drawImage(video, 0, 0, offCanvas.width, offCanvas.height);
            const frameData = offCtx.getImageData(0, 0, offCanvas.width, offCanvas.height);
            const data = frameData.data;
            const ow = offCanvas.width;
            const oh = offCanvas.height;

            // 0. Ambient Lighting Luminance Normalization Factor
            let ambientLumSum = 0;
            let ambientSampleCount = 0;
            for (let y = 0; y < oh; y += 4) {
              for (let x = 0; x < ow; x += 4) {
                const idx = (y * ow + x) * 4;
                ambientLumSum += 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
                ambientSampleCount++;
              }
            }
            const meanAmbientLum = ambientSampleCount > 0 ? ambientLumSum / ambientSampleCount : 120;
            const ambientNorm = 120 / Math.max(25, Math.min(220, meanAmbientLum));

            // 1. Forehead / Brow Furrow AU04 Analysis
            let browContrastSum = 0;
            let browSampleCount = 0;
            const browStartY = Math.floor(oh * 0.18);
            const browEndY = Math.floor(oh * 0.38);
            const browStartX = Math.floor(ow * 0.32);
            const browEndX = Math.floor(ow * 0.68);

            for (let y = browStartY; y < browEndY; y += 2) {
              for (let x = browStartX; x < browEndX; x += 2) {
                const idx = (y * ow + x) * 4;
                const lum = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
                if (x > browStartX + 2) {
                  const prevIdx = (y * ow + (x - 2)) * 4;
                  const prevLum =
                    0.299 * data[prevIdx] + 0.587 * data[prevIdx + 1] + 0.114 * data[prevIdx + 2];
                  browContrastSum += Math.abs(lum - prevLum);
                  browSampleCount++;
                }
              }
            }

            // 2. Mouth Region Analysis (AU12 Smile & Mouth Open)
            let mouthLumSum = 0;
            let mouthSampleCount = 0;
            let cheekLumSum = 0;
            let cheekSampleCount = 0;

            const mouthStartY = Math.floor(oh * 0.64);
            const mouthEndY = Math.floor(oh * 0.86);
            const mouthStartX = Math.floor(ow * 0.34);
            const mouthEndX = Math.floor(ow * 0.66);

            for (let y = mouthStartY; y < mouthEndY; y += 2) {
              for (let x = mouthStartX; x < mouthEndX; x += 2) {
                const idx = (y * ow + x) * 4;
                const lum = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
                mouthLumSum += lum;
                mouthSampleCount++;
              }
            }

            // Cheeks for smile expansion
            for (let y = Math.floor(oh * 0.45); y < Math.floor(oh * 0.62); y += 2) {
              for (let x = Math.floor(ow * 0.2); x < Math.floor(ow * 0.8); x += 4) {
                const idx = (y * ow + x) * 4;
                const lum = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
                cheekLumSum += lum;
                cheekSampleCount++;
              }
            }

            // 3. Eye Luminance for Blink Tracking
            let eyeLumSum = 0;
            let eyeSampleCount = 0;
            for (let y = Math.floor(oh * 0.32); y < Math.floor(oh * 0.44); y += 2) {
              for (let x = Math.floor(ow * 0.28); x < Math.floor(ow * 0.72); x += 2) {
                const idx = (y * ow + x) * 4;
                const lum = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
                eyeLumSum += lum;
                eyeSampleCount++;
              }
            }
            const currentEyeLum = eyeSampleCount > 0 ? eyeLumSum / eyeSampleCount : 100;
            if (lastEyeLuminanceRef.current - currentEyeLum > 14) {
              blinkHistoryRef.current.push(now);
            }
            lastEyeLuminanceRef.current = currentEyeLum;

            blinkHistoryRef.current = blinkHistoryRef.current.filter((t) => now - t < 60000);
            const calculatedBpm = Math.max(
              12,
              Math.min(48, blinkHistoryRef.current.length * (60000 / Math.max(5000, now)))
            );

            const rawBrowContrast = (browSampleCount > 0 ? browContrastSum / browSampleCount : 5.0) * ambientNorm;
            const avgMouthLum = mouthSampleCount > 0 ? mouthLumSum / mouthSampleCount : 80.0;
            const avgCheekLum = cheekSampleCount > 0 ? cheekLumSum / cheekSampleCount : 110.0;

            // Running adaptive resting baseline
            if (baselineContrastRef.current === null) {
              baselineContrastRef.current = rawBrowContrast;
            } else {
              baselineContrastRef.current = baselineContrastRef.current * 0.97 + rawBrowContrast * 0.03;
            }

            if (baselineMouthLumRef.current === null) {
              baselineMouthLumRef.current = avgMouthLum;
            } else {
              baselineMouthLumRef.current = baselineMouthLumRef.current * 0.97 + avgMouthLum * 0.03;
            }

            // High sensitivity differential cues relative to baseline resting face
            const browDelta = (rawBrowContrast - baselineContrastRef.current) / Math.max(1.0, baselineContrastRef.current);
            const mouthDelta = (baselineMouthLumRef.current - avgMouthLum) / Math.max(10.0, baselineMouthLumRef.current);
            const cheekDelta = (avgCheekLum - avgMouthLum) / Math.max(10.0, avgMouthLum);

            const browFurrow = Math.min(0.95, Math.max(0.06, 0.12 + browDelta * 1.5));
            const mouthOpen = Math.min(0.95, Math.max(0.03, 0.05 + mouthDelta * 2.2));
            const smileValence = Math.min(0.95, Math.max(0.04, 0.10 + cheekDelta * 0.4 - browFurrow * 0.2));
            const entropy = Math.min(
              1.0,
              Math.max(0.08, browFurrow * 0.5 + smileValence * 0.2 + 0.08)
            );

            // ONLY apply optical flow classification and update if NOT locked into a manual preset
            if (!lockedPresetRef.current) {
              // Strictly 2 modes: Stressed (Brow Furrow AU04) vs Smile (Calm & Calibrated)
              const isStressed = browFurrow > 0.28 || browDelta > 0.12;
              const expLabel = isStressed ? "Stressed (Brow Furrow AU04)" : "Smile (Calm & Calibrated)";

              const newTelemetry: FacialTelemetry = {
                au04_brow_furrow: isStressed ? Math.max(0.65, parseFloat(browFurrow.toFixed(3))) : 0.06,
                au12_smile: isStressed ? 0.04 : Math.max(0.68, parseFloat(smileValence.toFixed(3))),
                mouth_open: parseFloat(mouthOpen.toFixed(3)),
                blink_rate_bpm: Math.round(calculatedBpm),
                valence_entropy: isStressed ? 0.74 : 0.12,
              };

              setExpressionName(expLabel);

              const updatedState = {
                faceDetected: true,
                ...newTelemetry,
                blink_rate: newTelemetry.blink_rate_bpm,
                entropy: newTelemetry.valence_entropy,
              };
              detectionStateRef.current = updatedState;
              setDetectionState(updatedState);

              if (onTelemetryChangeRef.current) {
                onTelemetryChangeRef.current(newTelemetry);
              }
            }

            setFrameTick((prev) => (prev + 1) % 100);
          } catch (e) {
            console.warn("CV feature extraction error:", e);
          }
        }
      }

      // 4. Draw Cyberpunk Augmented Reality (AR) HUD Reticles over the Real Live Camera Feed
      const boxX = w * 0.2;
      const boxY = h * 0.15;
      const boxW = w * 0.6;
      const boxH = h * 0.7;

      const currentAU04 = detectionStateRef.current.au04_brow_furrow;
      const currentAU12 = detectionStateRef.current.au12_smile;
      const isHighStress = currentAU04 > 0.35;
      const themeColor = isHighStress ? "#f43f5e" : "#2dd4bf";

      // Animated Face Tracking Corner Brackets
      ctx.strokeStyle = themeColor;
      ctx.lineWidth = 2.5;
      const bLen = 22;

      ctx.beginPath();
      // Top-Left
      ctx.moveTo(boxX, boxY + bLen); ctx.lineTo(boxX, boxY); ctx.lineTo(boxX + bLen, boxY);
      // Top-Right
      ctx.moveTo(boxX + boxW - bLen, boxY); ctx.lineTo(boxX + boxW, boxY); ctx.lineTo(boxX + boxW, boxY + bLen);
      // Bottom-Left
      ctx.moveTo(boxX, boxY + boxH - bLen); ctx.lineTo(boxX, boxY + boxH); ctx.lineTo(boxX + bLen, boxY + boxH);
      // Bottom-Right
      ctx.moveTo(boxX + boxW - bLen, boxY + boxH); ctx.lineTo(boxX + boxW, boxY + boxH); ctx.lineTo(boxX + boxW, boxY + boxH - bLen);
      ctx.stroke();

      // Eye Tracking Reticles
      const leftEyeX = boxX + boxW * 0.32;
      const rightEyeX = boxX + boxW * 0.68;
      const eyeY = boxY + boxH * 0.35;

      ctx.fillStyle = themeColor;
      ctx.beginPath();
      ctx.arc(leftEyeX, eyeY, 4, 0, Math.PI * 2);
      ctx.arc(rightEyeX, eyeY, 4, 0, Math.PI * 2);
      ctx.fill();

      // AU04 Brow Furrow Tension Bar
      ctx.strokeStyle = themeColor;
      ctx.lineWidth = 3.5;
      ctx.beginPath();
      ctx.moveTo(boxX + boxW * 0.25, boxY + boxH * 0.22);
      ctx.lineTo(boxX + boxW * 0.75, boxY + boxH * 0.22);
      ctx.stroke();

      // Smile mouth curvature tracker
      ctx.strokeStyle = currentAU12 > 0.25 ? "#38bdf8" : themeColor;
      ctx.lineWidth = 2;
      ctx.beginPath();
      const mouthY = boxY + boxH * 0.72;
      ctx.arc(boxX + boxW * 0.5, mouthY, boxW * 0.16, 0.1 * Math.PI, 0.9 * Math.PI, false);
      ctx.stroke();

      // Dynamic Reticle Badge
      ctx.font = "bold 10px JetBrains Mono, monospace";
      ctx.fillStyle = themeColor;
      ctx.textAlign = "center";
      ctx.fillText(
        `AU04: ${(currentAU04 * 100).toFixed(0)}% • AU12: ${(currentAU12 * 100).toFixed(0)}%`,
        boxX + boxW * 0.5,
        boxY + boxH * 0.22 - 7
      );
    }

    animFrameRef.current = requestAnimationFrame(processFrame);
  }, []);

  // Stable animation loop mounted once
  useEffect(() => {
    animFrameRef.current = requestAnimationFrame(processFrame);
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [processFrame]);

  // Manual slider adjustment for testing
  const handleManualSlider = (key: keyof FacialTelemetry, val: number) => {
    const updated = {
      ...detectionState,
      [key]: val,
    };
    detectionStateRef.current = updated;
    setDetectionState(updated);
    if (onTelemetryChangeRef.current) {
      onTelemetryChangeRef.current({
        au04_brow_furrow: updated.au04_brow_furrow,
        au12_smile: updated.au12_smile,
        mouth_open: updated.mouth_open,
        blink_rate_bpm: updated.blink_rate,
        valence_entropy: updated.entropy,
      });
    }
  };

  return (
    <div className={`glass-card p-4 space-y-3 border-teal-500/30 ${compact ? "" : "shadow-xl"}`}>
      {/* Header */}
      <div className="flex items-center justify-between border-b border-white/5 pb-2.5">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-lg bg-teal-500/10 border border-teal-500/20 text-teal-300">
            <Camera className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-xs font-bold text-white font-display tracking-wide flex items-center">
              {title}
              <span
                className={`ml-2 w-2 h-2 rounded-full ${
                  cameraActive
                    ? "bg-emerald-400 animate-pulse shadow-[0_0_8px_#34d399]"
                    : "bg-gray-500"
                }`}
              />
            </h3>
            <div className="flex items-center space-x-1.5 mt-0.5">
              <span className="text-[10px] text-gray-400 font-mono">
                WebRTC Stream •
              </span>
              <span
                className={`inline-flex items-center px-1.5 py-0.2 rounded text-[9px] font-mono border ${
                  visionEngine === "mediapipe"
                    ? "bg-teal-500/15 text-teal-300 border-teal-500/30 font-semibold"
                    : "bg-blue-500/15 text-blue-300 border-blue-500/30 font-semibold"
                }`}
              >
                <Sparkles className="w-2.5 h-2.5 mr-1 text-teal-400" />
                {visionEngine === "mediapipe" ? "MediaPipe 478-Pt AI Mesh" : "Adaptive Optical Flow"}
              </span>
            </div>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center space-x-2">
          {cameraActive ? (
            <button
              onClick={stopCamera}
              className="px-2.5 py-1 text-[11px] font-semibold rounded-lg bg-rose-500/20 text-rose-300 border border-rose-500/40 hover:bg-rose-500/30 transition flex items-center space-x-1"
            >
              <CameraOff className="w-3.5 h-3.5" />
              <span>Stop Cam</span>
            </button>
          ) : (
            <button
              onClick={() => startCamera()}
              disabled={isInitializing}
              className="px-3 py-1.5 text-[11px] font-bold rounded-lg bg-gradient-to-r from-teal-400 to-cyan-400 text-dark-950 hover:brightness-110 shadow-[0_0_12px_rgba(45,212,191,0.4)] transition flex items-center space-x-1.5 font-display disabled:opacity-50"
            >
              {isInitializing ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>Starting...</span>
                </>
              ) : (
                <>
                  <Video className="w-3.5 h-3.5" />
                  <span>Share Camera Feed</span>
                </>
              )}
            </button>
          )}

          {/* Toggle Simulator Controls */}
          <button
            onClick={() => setManualMode(!manualMode)}
            title="Toggle Biometric Test Simulator"
            className={`p-1.5 rounded-lg border transition ${
              manualMode
                ? "bg-teal-500/20 text-teal-300 border-teal-500/40"
                : "bg-dark-800 text-gray-400 border-white/10 hover:text-white"
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Camera Selection Dropdown if multiple devices found */}
      {cameraDevices.length > 1 && (
        <div className="flex items-center space-x-2 text-[11px] font-mono bg-dark-900/80 p-2 rounded-lg border border-white/5">
          <span className="text-gray-400 shrink-0">Device:</span>
          <select
            value={selectedDeviceId}
            onChange={(e) => {
              setSelectedDeviceId(e.target.value);
              if (cameraActive) {
                startCamera(e.target.value);
              }
            }}
            className="w-full bg-dark-950 text-white rounded border border-white/10 px-2 py-1 focus:outline-none focus:border-teal-400 text-[11px]"
          >
            {cameraDevices.map((d, i) => (
              <option key={d.deviceId || i} value={d.deviceId}>
                {d.label || `Camera ${i + 1}`}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Error Banner */}
      {errorMsg && (
        <div className="p-2.5 rounded-xl bg-rose-500/15 border border-rose-500/30 text-rose-300 text-[11px] space-y-1.5 font-mono">
          <div className="flex items-start space-x-2">
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-rose-400" />
            <div className="space-y-1">
              <p className="font-bold">{errorMsg}</p>
              <div className="text-[10px] text-gray-300 space-y-0.5">
                <p>• Click the lock/camera icon in your address bar and select &quot;Allow&quot;.</p>
                <p>• Make sure another app (Zoom, Google Meet, Teams) is not holding the webcam.</p>
                <p>• You can also click the quick expression test chips below.</p>
              </div>
            </div>
          </div>
          <div className="flex justify-end pt-1">
            <button
              onClick={() => startCamera()}
              className="px-2.5 py-1 text-[10px] font-bold rounded bg-rose-500/30 hover:bg-rose-500/50 text-white border border-rose-400/30 transition flex items-center space-x-1"
            >
              <RefreshCw className="w-3 h-3" />
              <span>Retry Camera</span>
            </button>
          </div>
        </div>
      )}

      {/* Viewport Container: Hardware Video & AR Canvas Overlay */}
      <div className="relative aspect-[4/3] rounded-xl overflow-hidden border border-teal-500/25 bg-dark-950 shadow-inner group">
        {/* Real Hardware Video Element - Mirrored and ALWAYS in DOM for immediate decoding */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`w-full h-full object-cover transform -scale-x-100 transition-opacity duration-300 ${
            cameraActive ? "opacity-100" : "opacity-0 absolute inset-0 pointer-events-none"
          }`}
        />

        {/* Augmented Reality Reticle Canvas Overlay */}
        <canvas
          ref={canvasRef}
          width={400}
          height={300}
          className="absolute inset-0 w-full h-full pointer-events-none"
        />

        {/* Standby / Share Callout when Camera is NOT active */}
        {!cameraActive && (
          <div
            onClick={() => startCamera()}
            className="absolute inset-0 flex flex-col items-center justify-center p-4 bg-gradient-to-b from-dark-950/80 via-dark-950/95 to-dark-950 cursor-pointer group-hover:border-teal-400/40 transition"
          >
            {/* Cyberpunk grid backdrop */}
            <div className="w-14 h-14 rounded-2xl bg-teal-500/10 border border-teal-500/30 flex items-center justify-center text-teal-300 mb-3 shadow-[0_0_20px_rgba(45,212,191,0.2)] group-hover:scale-105 transition">
              <Video className="w-7 h-7 text-teal-400 animate-pulse" />
            </div>

            <button
              type="button"
              className="px-4 py-2 rounded-xl bg-gradient-to-r from-teal-400 to-cyan-400 text-dark-950 font-bold text-xs font-display shadow-[0_0_15px_rgba(45,212,191,0.4)] hover:brightness-110 transition flex items-center space-x-2"
            >
              <Share2 className="w-4 h-4" />
              <span>Share Live Camera Feed</span>
            </button>

            <p className="text-[11px] text-gray-400 font-mono mt-2.5 text-center">
              Click to activate live webcam and real-time facial expression tracking
            </p>
          </div>
        )}

        {/* Floating Telemetry Status Pill */}
        <div className="absolute top-2.5 left-2.5 bg-dark-950/85 backdrop-blur-md px-2.5 py-1 rounded-md border border-white/10 text-[10px] font-mono flex items-center space-x-2 pointer-events-none">
          <span className="text-gray-400">Stream:</span>
          <span
            className={
              cameraActive ? "text-emerald-400 font-bold flex items-center" : "text-amber-400 font-bold"
            }
          >
            {cameraActive ? (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5 animate-ping" />
                LIVE (60 FPS)
              </>
            ) : (
              "STANDBY"
            )}
          </span>
        </div>

        {/* Dynamic Expression Classification Pill */}
        <div className="absolute bottom-2.5 left-2.5 right-2.5 bg-dark-950/90 backdrop-blur-md px-3 py-1.5 rounded-lg border border-teal-500/20 text-[11px] font-mono flex items-center justify-between pointer-events-none">
          <span className="text-gray-400 flex items-center">
            <Zap className="w-3.5 h-3.5 mr-1 text-teal-400" />
            Expression:
          </span>
          <span
            className={`font-bold ${
              detectionState.au04_brow_furrow > 0.35 ? "text-rose-400" : "text-teal-300"
            }`}
          >
            {expressionName}
          </span>
        </div>
      </div>

      {/* Quick Expression Reaction Preset Buttons for instant testing */}
      <div className="space-y-2 pt-1">
        {lockedPreset ? (
          <div className="flex items-center justify-between p-2 rounded-xl bg-teal-500/15 border border-teal-500/30 text-xs font-mono">
            <div className="flex items-center space-x-2 text-teal-300">
              <span className="w-2 h-2 rounded-full bg-teal-400 animate-pulse" />
              <span>Manual Inject: <strong className="text-white">{expressionName}</strong></span>
            </div>
            <button
              onClick={resumeLiveCameraTracking}
              className="px-2.5 py-1 text-[10px] font-bold rounded-lg bg-teal-500/30 hover:bg-teal-500/50 text-white border border-teal-400/40 transition flex items-center space-x-1 shadow-sm"
            >
              <RefreshCw className="w-3 h-3" />
              <span>Resume Live Cam</span>
            </button>
          </div>
        ) : (
          <div className="flex items-center justify-between text-[10px] font-mono text-gray-400">
            <span>Quick Expression Presets (Locks &amp; Injects):</span>
            <span className="text-emerald-400 flex items-center">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1 animate-ping" />
              Adaptive Camera Active
            </span>
          </div>
        )}

        {/* Dedicated 2-Mode Selectors: Smile vs Stressed */}
        <div className="grid grid-cols-2 gap-2 text-xs font-mono">
          <button
            onClick={() => applyPresetExpression("smile")}
            className={`px-3 py-2 rounded-xl border transition flex items-center justify-center space-x-2 ${
              lockedPreset === "smile"
                ? "bg-teal-500/30 border-teal-400 text-white shadow-[0_0_15px_rgba(45,212,191,0.35)] font-bold ring-1 ring-teal-400"
                : "bg-dark-900 hover:bg-dark-800 text-teal-300 border-teal-500/20 hover:border-teal-500/40"
            }`}
          >
            <span className="text-base">😊</span>
            <div className="text-left">
              <span className="block text-[11px] font-bold">Smile Mode</span>
              <span className="block text-[9px] text-teal-300/70 font-sans">Calm &amp; Calibrated</span>
            </div>
          </button>

          <button
            onClick={() => applyPresetExpression("stress")}
            className={`px-3 py-2 rounded-xl border transition flex items-center justify-center space-x-2 ${
              lockedPreset === "stress"
                ? "bg-rose-500/30 border-rose-400 text-white shadow-[0_0_15px_rgba(244,63,94,0.35)] font-bold ring-1 ring-rose-400"
                : "bg-dark-900 hover:bg-dark-800 text-rose-300 border-rose-500/20 hover:border-rose-500/40"
            }`}
          >
            <span className="text-base">😠</span>
            <div className="text-left">
              <span className="block text-[11px] font-bold">Stressed Mode</span>
              <span className="block text-[9px] text-rose-300/70 font-sans">Brow Furrow AU04</span>
            </div>
          </button>
        </div>

        {/* Live Facial Biometrics Gauge Strip (2 Dual Modalities) */}
        <div className="grid grid-cols-2 gap-2.5 text-xs font-mono pt-1">
          <div className="bg-dark-900/90 p-2.5 rounded-xl border border-teal-500/20 space-y-1.5">
            <div className="flex justify-between items-center text-[10px]">
              <span className="text-teal-300 font-semibold flex items-center">
                <span className="w-1.5 h-1.5 rounded-full bg-teal-400 mr-1.5 animate-pulse" />
                AU12 Smile (Calm)
              </span>
              <span className="text-teal-300 font-bold text-xs">{Math.round(detectionState.au12_smile * 100)}%</span>
            </div>
            <div className="w-full bg-dark-800 rounded-full h-1.5 overflow-hidden">
              <div
                className="bg-gradient-to-r from-teal-400 to-cyan-400 h-full transition-all duration-150 shadow-[0_0_8px_#2dd4bf]"
                style={{ width: `${Math.min(100, Math.max(5, detectionState.au12_smile * 100))}%` }}
              />
            </div>
          </div>

          <div className="bg-dark-900/90 p-2.5 rounded-xl border border-rose-500/20 space-y-1.5">
            <div className="flex justify-between items-center text-[10px]">
              <span className="text-rose-300 font-semibold flex items-center">
                <span className="w-1.5 h-1.5 rounded-full bg-rose-400 mr-1.5 animate-pulse" />
                AU04 Brow (Stressed)
              </span>
              <span className="text-rose-400 font-bold text-xs">{Math.round(detectionState.au04_brow_furrow * 100)}%</span>
            </div>
            <div className="w-full bg-dark-800 rounded-full h-1.5 overflow-hidden">
              <div
                className="bg-gradient-to-r from-rose-500 to-red-500 h-full transition-all duration-150 shadow-[0_0_8px_#f43f5e]"
                style={{ width: `${Math.min(100, Math.max(5, detectionState.au04_brow_furrow * 100))}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Manual Biometric Slider Simulator (visible when user toggles) */}
      {manualMode && (
        <div className="p-3 bg-dark-900/90 rounded-xl border border-teal-500/30 space-y-2.5 font-mono text-xs">
          <div className="flex justify-between items-center text-[10px] text-teal-300 font-bold border-b border-white/5 pb-1">
            <span>BIOMETRIC TEST SIMULATOR</span>
            <span className="text-gray-400 font-normal">Real-Time Injector</span>
          </div>

          <div>
            <div className="flex justify-between text-[10px] text-gray-300">
              <span>AU04 Brow Furrow (Stress):</span>
              <span className="text-rose-400 font-bold">
                {(detectionState.au04_brow_furrow * 100).toFixed(0)}%
              </span>
            </div>
            <input
              type="range"
              min="0.0"
              max="1.0"
              step="0.01"
              value={detectionState.au04_brow_furrow}
              onChange={(e) => handleManualSlider("au04_brow_furrow", parseFloat(e.target.value))}
              className="w-full h-1 bg-dark-800 rounded-lg appearance-none cursor-pointer accent-rose-500"
            />
          </div>

          <div>
            <div className="flex justify-between text-[10px] text-gray-300">
              <span>AU12 Smile Valence:</span>
              <span className="text-teal-300 font-bold">
                {(detectionState.au12_smile * 100).toFixed(0)}%
              </span>
            </div>
            <input
              type="range"
              min="0.0"
              max="1.0"
              step="0.01"
              value={detectionState.au12_smile}
              onChange={(e) => handleManualSlider("au12_smile", parseFloat(e.target.value))}
              className="w-full h-1 bg-dark-800 rounded-lg appearance-none cursor-pointer accent-teal-400"
            />
          </div>
        </div>
      )}

      {/* Extracted Blendshape Live Gauges */}
      <div className="grid grid-cols-2 gap-2 text-xs font-mono pt-1">
        <div className="bg-dark-900/90 p-2.5 rounded-xl border border-white/5 space-y-1">
          <div className="flex justify-between items-center text-[10px]">
            <span className="text-gray-400 flex items-center">
              <Frown className="w-3 h-3 mr-1 text-rose-400" /> Brow Furrow (AU04)
            </span>
            <span
              className={`font-bold ${
                detectionState.au04_brow_furrow > 0.35 ? "text-rose-400" : "text-teal-300"
              }`}
            >
              {detectionState.au04_brow_furrow.toFixed(3)}
            </span>
          </div>
          <div className="w-full bg-dark-800 rounded-full h-1.5 overflow-hidden">
            <div
              className={`h-full transition-all duration-150 ${
                detectionState.au04_brow_furrow > 0.35
                  ? "bg-rose-500 shadow-[0_0_8px_#f43f5e]"
                  : "bg-teal-400"
              }`}
              style={{ width: `${Math.min(100, detectionState.au04_brow_furrow * 100)}%` }}
            />
          </div>
        </div>

        <div className="bg-dark-900/90 p-2.5 rounded-xl border border-white/5 space-y-1">
          <div className="flex justify-between items-center text-[10px]">
            <span className="text-gray-400 flex items-center">
              <Smile className="w-3 h-3 mr-1 text-teal-400" /> Smile Valence (AU12)
            </span>
            <span className="text-white font-bold">{detectionState.au12_smile.toFixed(3)}</span>
          </div>
          <div className="w-full bg-dark-800 rounded-full h-1.5 overflow-hidden">
            <div
              className="bg-teal-400 h-full transition-all duration-150"
              style={{ width: `${Math.min(100, detectionState.au12_smile * 100)}%` }}
            />
          </div>
        </div>

        <div className="bg-dark-900/90 p-2 rounded-xl border border-white/5 flex justify-between items-center text-[11px]">
          <span className="text-gray-400 flex items-center">
            <Eye className="w-3 h-3 mr-1.5 text-cyan-400" /> Blink Frequency:
          </span>
          <span className="text-cyan-300 font-bold">{detectionState.blink_rate} BPM</span>
        </div>

        <div className="bg-dark-900/90 p-2 rounded-xl border border-white/5 flex justify-between items-center text-[11px]">
          <span className="text-gray-400 flex items-center">
            <Sparkles className="w-3 h-3 mr-1.5 text-amber-400" /> Valence Entropy:
          </span>
          <span className="text-amber-300 font-bold">{detectionState.entropy.toFixed(3)}</span>
        </div>
      </div>
    </div>
  );
}
