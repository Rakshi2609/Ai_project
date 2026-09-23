import os
import pytest
import numpy as np
from emotion_webcam import load_emotion_keras_model, predict_emotion_from_face, EMOTIONS

def test_emotion_model_loading_and_prediction():
    model_path = "emotion_model.h5"
    assert os.path.exists(model_path), f"{model_path} should exist in project root!"
    
    # Load model
    model = load_emotion_keras_model(model_path)
    assert model is not None
    
    # Create dummy 48x48 grayscale face crop
    dummy_face = np.random.randint(0, 256, (48, 48), dtype=np.uint8)
    
    emotion, confidence, probs = predict_emotion_from_face(dummy_face, model)
    
    assert emotion in EMOTIONS
    assert 0.0 <= confidence <= 1.0
    assert len(probs) == 7
    assert np.isclose(np.sum(probs), 1.0, atol=1e-3)
