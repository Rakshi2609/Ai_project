export interface FacialTelemetry {
  au04_brow_furrow: number; // 0.0 - 1.0 (brow furrow / stress / frown)
  blink_rate_bpm: number; // 10 - 45 (eye blink frequency)
  au12_smile: number; // 0.0 - 1.0 (lip corner pull / smile / calm)
  mouth_open: number; // 0.0 - 1.0 (speech / startle)
  valence_entropy: number; // 0.0 - 1.0 (micro-expression volatility)
}

export interface VocalTelemetry {
  pitch_mean_hz: number; // e.g. 100 - 350 Hz
  acoustic_jitter_pct: number; // 0.2 - 4.5%
  intensity_db: number; // 30 - 85 dB
  pause_ratio: number; // 0.0 - 0.6
  speech_duration_s: number;
}

export interface PhysioTelemetry {
  bvp_pulse_rate_bpm: number;
  eda_skin_conductance_us: number;
  respiration_rate_bpm: number;
  has_motion_artifact: boolean;
}

export interface RobotState {
  mode: 'correct' | 'wrong';
  error_type: 'nominal' | 'gripper_slip' | 'trajectory_drift' | 'excessive_speed' | 'torque_anomaly';
  speed_mps: number;
  drift_m: number;
  torque_anomaly: number;
}

export interface InferenceResponse {
  status: string;
  module_1_telemetry: {
    robot_kinematics: {
      actual_speed_mps: number;
      normalized_drift: number;
      error_type: string;
      joint_torque_anomaly: number;
      kinematic_reliability_score: number;
    };
    facial_blendshapes: {
      au04_brow_furrow: number;
      valence_entropy: number;
      facial_stress_index: number;
      calm_score: number;
    };
    vocal_acoustics: {
      f0_mean_hz: number;
      jitter_pct: number;
      snr_db: number;
      vocal_stress_index: number;
    };
    physiological_bvp: {
      pulse_rate_bpm: number;
      eda_microsiemens: number;
      has_motion_artifact: boolean;
    };
  };
  module_3_cross_modal_attention: {
    dynamic_attention_weights: {
      robot_kinematics: number;
      facial_affect: number;
      vocal_acoustics: number;
      physiological_bvp: number;
    };
    noise_attenuation_applied: boolean;
  };
  module_4_trust_prediction: {
    continuous_trust_score: number;
    trust_state: 'CALIBRATED_TRUST' | 'UNDER_TRUST' | 'OVER_TRUST';
    state_description: string;
    prediction_uncertainty: number;
    latency_ms?: number;
    decision_latency_ms?: number;
    system_confidence?: number;
    latency_compliant_250ms?: boolean;
  };
  module_5_mitigation_policy: {
    recommended_action: string;
    execution_speed_factor: number;
    requires_operator_validation: boolean;
    hud_transparency_message: string;
    low_level_control_signal: string;
  };
}

export interface SupervisorFeedback {
  session_id: string;
  task_name: string;
  supervisor_ground_truth_trust: number;
  subject_id: string;
  notes?: string;
}

export function calculateCognitiveStress(face: FacialTelemetry, voice?: VocalTelemetry): number {
  const browStress = face.au04_brow_furrow * 50;
  const entropyStress = face.valence_entropy * 25;
  const voiceStress = voice ? (voice.acoustic_jitter_pct / 3.0) * 25 : 10;
  return Math.min(100, Math.round(browStress + entropyStress + voiceStress));
}

export function classifyTrustZone(trustScore: number): 'UNDER_TRUST' | 'CALIBRATED_TRUST' | 'OVER_TRUST' {
  if (trustScore < 0.35) return 'UNDER_TRUST';
  if (trustScore > 0.75) return 'OVER_TRUST';
  return 'CALIBRATED_TRUST';
}

