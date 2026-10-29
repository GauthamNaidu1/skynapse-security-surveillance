"""
=============================================================================
RESEARCH & CAPABILITY DISCLOSURE
=============================================================================
This script utilizes an off-the-shelf, pre-trained Google MoViNet-A0 checkpoint 
trained on the Kinetics-600 human action video dataset.

CRITICAL DISCLOSURES:
1. NO MODEL TRAINING: This model was NOT trained or fine-tuned by the author.
   Custom CCTV crime action training did not occur.
2. NO BENCHMARK EVALUATION: No empirical accuracy, precision, recall, or loss
   metrics were measured on crime video benchmarks.
3. APPROXIMATE DEMO MAPPING: The index mapping to crime categories (e.g., indices
   150, 400, 500) is a simplified demonstration heuristic, not a trained classifier.
4. ZERO-SHOT ALTERNATIVE: The application also supports Gemini (gemini-3.1-pro-preview)
   via API-based zero-shot prompt classification over extracted video frames.
=============================================================================
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import json
import argparse
import sys

def load_labels():
    """
    Stub notice: Full Kinetics-600 label resolution is omitted in this demo.
    Action classifications are approximate demonstration mappings.
    """
    pass

class VideoAnalyzer:
    def __init__(self, model_path):
        print(f"Loading MoViNet model from {model_path}...", file=sys.stderr)
        self.model = tf.saved_model.load(model_path)
        self.infer = self.model.signatures['serving_default']
        
    def preprocess_video(self, video_path, num_frames=32):
        cap = cv2.VideoCapture(video_path)
        frames = []
        count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip = max(1, total_frames // num_frames)
        
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if count % skip == 0:
                frame = cv2.resize(frame, (172, 172))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            count += 1
        cap.release()
        
        while len(frames) < num_frames:
            frames.append(np.zeros((172, 172, 3), dtype=np.uint8))
            
        video = np.array(frames, dtype=np.float32)
        video = video / 255.0
        return np.expand_dims(video, axis=0) # [1, 32, 172, 172, 3]

    def analyze(self, video_path):
        video_input = self.preprocess_video(video_path)
        # MoViNet a0-base takes [1, frames, 172, 172, 3]
        # Based on inspection: input is 'image', output is 'classifier_head'
        outputs = self.infer(image=tf.constant(video_input))
        
        probs = outputs['classifier_head'].numpy()[0]
        
        # In real use, you'd use kinetics-600 labels
        # Here we map top indices to possible crime types for demonstration
        top_idx = np.argmax(probs)
        confidence = float(probs[top_idx])
        
        # Simplified mapping logic for demo
        detected_type = "Normal"
        if confidence > 0.3:
            # Indices for common actions in Kinetics-600 (approximate for demo)
            # In a full impl, we'd use the actual labels.csv
            indices_map = {
                150: "Fighting",    # Approximate index for fighting
                400: "Theft",       # Approximate index for picking pocket
                500: "Vandalism",   # Approximate index for graffiti
            }
            detected_type = indices_map.get(top_idx, "Activity Detected")
            if detected_type == "Activity Detected" and confidence < 0.5:
                detected_type = "Normal"
            
        return {
            "detected": detected_type != "Normal",
            "type": detected_type,
            "confidence": confidence,
            "details": f"MoViNet detected action signatures: {detected_type}.",
            "reasoning": f"Inference matched Kinetics-600 index {top_idx} with {confidence*100:.1f}% confidence."
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    
    analyzer = VideoAnalyzer(args.model)
    result = analyzer.analyze(args.video)
    print(json.dumps(result))
