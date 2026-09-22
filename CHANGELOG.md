# Changelog

All notable changes to the Universal Robots UR5 Multimodal Continuous Human-Robot Trust Calibration System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.0.0] - 2026-09-22

### Added
- **Multi-Route WebRTC Persistence:** Live camera and audio streaming verification across all 4 production routes (`/`, `/capture`, `/robot`, `/analytics`) with session storage memory.
- **Audio VU Meter Indicator:** Real-time visual decibel VU meter and dynamic waveform canvas in `VoiceCapture.tsx`.
- **UR5 6-DOF Kinematics Ribbon:** Real-time joint angle and Cartesian end-effector position telemetry feed in `Cobot3DView.tsx`.
- **Dynamic HUD Visual Risk Pulse:** Real-time radial dial pulse alerts, ISO/TS 15066 safety badges, and emergency dual-operator mitigation flags in `TrustEngineHUD.tsx`.
- **Ambient Lighting Contrast Normalizer:** Automatic global luminance baseline calibration in `FaceExpressionCapture.tsx` preventing AU04/AU12 drift in low-light environments.
- **Temporal Ring-Buffer Eviction:** Bounded sliding window cache in `trust_ai_pipeline.py` preventing memory growth under continuous high-frequency inference.
- **Automated Route Test Suite:** `tests/test_routes.py` verifying HTTP 200 OK, latency benchmarks, and payload validation across all system endpoints.
- **C4 Architecture Specification:** Comprehensive architectural diagrams and attention fusion mathematical formulation in `ARCHITECTURE.md`.
- **Hardware Diagnostics in Orchestrator:** `/dev/video*` hardware camera autodetection and automated test execution integrated into `run.sh`.

### Fixed
- **WebRTC / Web Audio Teardown Bug:** Fixed race condition where React state updates inside animation loops triggered track stops, locking the camera indicator light while freezing video frames and starving trust scoring.
- **Trust Score Freeze:** Resolved frozen dial state at 79% by ensuring continuous 11 Hz inference dispatches from live facial optical flow telemetry.

### Performance
- **Sub-2ms Inference Latency:** Mean edge inference latency maintained at $1.1\text{ ms}$ ($< 250\text{ ms}$ threshold).
- **LOSO Cross-Validation:** Calibrated trust prediction error bounds ($MSE < 0.08$) across cross-subject validation splits.
