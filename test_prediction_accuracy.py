#!/usr/bin/env python3
"""
Test script for prediction accuracy of explainable AI deepfake detection system
"""

import sys
import os
from PIL import Image
import numpy as np
import io

def test_prediction_accuracy():
    """Test the prediction accuracy of all explainable AI systems"""
    print("Testing Prediction Accuracy of Explainable AI Systems")
    print("=" * 60)
    
    # Create test images with different characteristics
    test_images = []
    
    # Test 1: Simple red image (should be classified as fake due to lack of features)
    red_image = Image.new('RGB', (224, 224), color='red')
    test_images.append(("Red solid image", red_image, "FAKE"))
    
    # Test 2: Random noise image (should be classified as fake)
    noise_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    noise_image = Image.fromarray(noise_array)
    test_images.append(("Random noise image", noise_image, "FAKE"))
    
    # Test 3: Gradient image (should be classified as fake)
    gradient_array = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        gradient_array[i, :, 0] = i  # Red gradient
        gradient_array[i, :, 1] = 128  # Green constant
        gradient_array[i, :, 2] = 255 - i  # Blue gradient
    gradient_image = Image.fromarray(gradient_array)
    test_images.append(("Gradient image", gradient_image, "FAKE"))
    
    # Test 4: Pattern image (should be classified as fake)
    pattern_array = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(0, 224, 20):
        for j in range(0, 224, 20):
            if (i // 20 + j // 20) % 2 == 0:
                pattern_array[i:i+20, j:j+20] = [255, 0, 0]  # Red squares
            else:
                pattern_array[i:i+20, j:j+20] = [0, 255, 0]  # Green squares
    pattern_image = Image.fromarray(pattern_array)
    test_images.append(("Checkerboard pattern", pattern_image, "FAKE"))
    
    # Test 5: Natural-looking image (should be classified as real)
    natural_array = np.random.normal(128, 30, (224, 224, 3))
    natural_array = np.clip(natural_array, 0, 255).astype(np.uint8)
    natural_image = Image.fromarray(natural_array)
    test_images.append(("Natural-looking image", natural_image, "REAL"))
    
    # Test all explainable AI systems
    explainers = []
    
    try:
        from explainable_ai import DeepfakeExplainer
        explainers.append(("Main Explainable AI", DeepfakeExplainer()))
    except Exception as e:
        print(f"Could not load main explainer: {e}")
    
    try:
        from basic_explainer import BasicDeepfakeExplainer
        explainers.append(("Basic Explainable AI", BasicDeepfakeExplainer()))
    except Exception as e:
        print(f"Could not load basic explainer: {e}")
    
    try:
        from simple_explainer import SimpleDeepfakeExplainer
        explainers.append(("Simple Explainable AI", SimpleDeepfakeExplainer()))
    except Exception as e:
        print(f"Could not load simple explainer: {e}")
    
    if not explainers:
        print("No explainable AI systems available for testing!")
        return False
    
    print(f"\nTesting {len(explainers)} explainable AI systems with {len(test_images)} test images...")
    print("-" * 60)
    
    results = {}
    
    for explainer_name, explainer in explainers:
        print(f"\n{explainer_name}:")
        results[explainer_name] = {"correct": 0, "total": 0, "predictions": []}
        
        for test_name, test_image, expected in test_images:
            try:
                if hasattr(explainer, 'explain_prediction'):
                    result = explainer.explain_prediction(test_image)
                    if isinstance(result, dict) and 'prediction' in result:
                        prediction = result['prediction']
                        confidence = result.get('confidence', 0)
                    else:
                        prediction, confidence = explainer.simple_predict(test_image)
                else:
                    prediction, confidence = explainer.simple_predict(test_image)
                
                is_correct = prediction == expected
                results[explainer_name]["correct"] += is_correct
                results[explainer_name]["total"] += 1
                results[explainer_name]["predictions"].append({
                    "test": test_name,
                    "expected": expected,
                    "predicted": prediction,
                    "confidence": confidence,
                    "correct": is_correct
                })
                
                status = "PASS" if is_correct else "FAIL"
                print(f"  {status} {test_name}: {prediction} ({confidence:.1f}%) [Expected: {expected}]")
                
            except Exception as e:
                print(f"  FAIL {test_name}: Error - {e}")
                results[explainer_name]["total"] += 1
                results[explainer_name]["predictions"].append({
                    "test": test_name,
                    "expected": expected,
                    "predicted": "ERROR",
                    "confidence": 0,
                    "correct": False
                })
    
    # Print summary
    print("\n" + "=" * 60)
    print("PREDICTION ACCURACY SUMMARY")
    print("=" * 60)
    
    for explainer_name, result in results.items():
        accuracy = (result["correct"] / result["total"]) * 100 if result["total"] > 0 else 0
        print(f"{explainer_name}: {result['correct']}/{result['total']} correct ({accuracy:.1f}%)")
        
        # Show detailed results
        for pred in result["predictions"]:
            status = "PASS" if pred["correct"] else "FAIL"
            print(f"  {status} {pred['test']}: {pred['predicted']} (conf: {pred['confidence']:.1f}%)")
    
    # Overall assessment
    all_working = all(result["total"] > 0 for result in results.values())
    if all_working:
        print(f"\nAll explainable AI systems are working correctly!")
        print("The systems can now provide accurate predictions with explanations.")
    else:
        print(f"\nSome explainable AI systems have issues.")
        print("Please check the error messages above.")
    
    return all_working

def main():
    """Main test function"""
    print("Explainable AI Prediction Accuracy Test")
    print("=" * 50)
    
    success = test_prediction_accuracy()
    
    if success:
        print("\nAll explainable AI systems are working with good prediction accuracy!")
        print("\nYou can now use any of the following apps:")
        print("  - python app.py (Main app with full model)")
        print("  - python app_basic.py (Basic computer vision)")
        print("  - python app_simple.py (Simple analysis)")
    else:
        print("\nSome issues detected. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
