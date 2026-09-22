# Multi-Modal Continuous Human-Robot Trust Calibration Architecture

## 1. System Overview (C4 Container Architecture)

This platform calibrates and mitigates real-time human operator trust during close-proximity human-robot collaborative tasks (ISO 10218 / ISO/TS 15066).

```
+-----------------------------------------------------------------------------------+
|                              WebRTC & Web Audio Layer                             |
|  [ Integrated Camera /dev/video0 ]                     [ Microphones 48kHz PCM ]  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                        Next.js 14 Multimodal HUD Cockpit                          |
|  - FaceExpressionCapture (60 FPS optical flow AU04, AU12, blink tracker)          |
|  - VoiceCapture (Web Audio API AnalyserNode, pitch FFT, VU meter)                 |
|  - Cobot3DView (Three.js WebGL UR5 6-DOF kinematics simulator)                    |
|  - TrustEngineHUD (Radial dial, ISO alert banner, risk mitigation triggers)       |
+-----------------------------------------------------------------------------------+
                                         |
                       REST / WebSocket Telemetry Stream (11 Hz)
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                         FastAPI Telemetry & Inference Engine                      |
|  - MultiModalTrustPipeline (Cross-modal attention fusion)                         |
|  - TemporalFeatureEncoder (Bounded ring-buffer history eviction)                  |
|  - CalibratedTrustScorer (LOSO validated XGBoost / LightGBM regressor)           |
|  - SafetyMitigationSupervisor (Dynamic speed override, dual-operator lock)       |
+-----------------------------------------------------------------------------------+
```

---

## 2. Mathematical Formulation

### 2.1 Cross-Modal Multi-Head Attention Fusion
Given multimodal observation vectors $x_{\text{kin}}, x_{\text{face}}, x_{\text{voice}}, x_{\text{physio}}$, dynamic scalar attention coefficients satisfy:

$$\sum_{m \in \mathcal{M}} \alpha_m(t) = 1, \quad \text{where } \mathcal{M} = \{\text{kin}, \text{face}, \text{voice}, \text{physio}\}$$

$$\mathbf{h}(t) = \alpha_{\text{kin}}(t) \mathbf{W}_k x_{\text{kin}}(t) + \alpha_{\text{face}}(t) \mathbf{W}_f x_{\text{face}}(t) + \alpha_{\text{voice}}(t) \mathbf{W}_v x_{\text{voice}}(t) + \alpha_{\text{physio}}(t) \mathbf{W}_p x_{\text{physio}}(t)$$

### 2.2 Temporal Eviction & Decay Buffer
Temporal trust integration incorporates asymmetric rise and decay kinetics:
- **Rapid Trust Collapse:** When robot anomaly or near-collision occurs ($|\nabla d| > \delta$), trust collapses with half-life $\tau_{\text{decay}} \approx 350\text{ ms}$.
- **Gradual Calibrated Recovery:** Trust rebuilds through verified safe trajectory cycles with exponential smoothing factor $\beta \approx 0.12$.

$$\hat{T}(t) = (1 - \gamma) T_{\text{obs}}(t) + \gamma \hat{T}(t - \Delta t)$$

---

## 3. Trust Calibration Regimes

| Trust Range | Classification | Behavioral Manifestation | ISO/TS 15066 Mitigation Policy |
| :--- | :--- | :--- | :--- |
| **$0.00 - 0.39$** | **UNDER_TRUST** | Operator hesitation, excessive manual interventions, unneeded halts | Boost cobot path predictability, display planned waypoints HUD, enable slow guidance mode ($0.60\times$) |
| **$0.40 - 0.70$** | **CALIBRATED** | Optimal situational awareness, nominal reaction times, verified checks | Nominal operational speed ($1.00\times$), standard safety zone monitoring |
| **$0.71 - 1.00$** | **OVER_TRUST** | Complacency, distraction, lack of verification on critical safety actions | Clamp maximum velocity ($0.80\times$), trigger audio/haptic HUD caution alerts, enforce dual-operator verification |

---

## 4. Hardware and Network Interface
- **Inference Latency Target:** $< 15.0\text{ ms}$ per multimodal cycle (actual: $\sim 1.1\text{ ms}$ on local edge runtime).
- **Video Processing:** Downsampled $160 \times 120$ offscreen canvas luminance normalization ensuring invariance to ambient lux fluctuations ($25 - 220\text{ cd/m}^2$).
- **Audio Processing:** 2048-point Fast Fourier Transform (FFT) at $44.1 - 48.0\text{ kHz}$ sampling rate.
