#!/usr/bin/env python3
"""
Test script for the Molecular Solubility Prediction API.
This script sends requests to the API endpoints and verifies the responses.
"""
import argparse
import requests
import json
import base64
from io import BytesIO
from PIL import Image
import sys
import time
import os

def test_api(base_url):
    """Test the API endpoints."""
    print(f"Testing API at {base_url}...")
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"❌ Health check failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed with error: {e}")
        return False
    
    # Test sample molecules endpoint
    print("\n2. Testing sample molecules endpoint...")
    try:
        response = requests.get(f"{base_url}/sample-molecules")
        if response.status_code == 200:
            data = response.json()
            if "samples" in data and len(data["samples"]) > 0:
                print(f"✅ Sample molecules endpoint passed. Found {len(data['samples'])} molecules")
                # Save a sample for later use
                sample_smiles = data["samples"][0]["smiles"]
                sample_name = data["samples"][0]["name"]
                print(f"Using sample molecule: {sample_name} ({sample_smiles})")
            else:
                print("❌ Sample molecules endpoint failed: No samples found")
                return False
        else:
            print(f"❌ Sample molecules endpoint failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Sample molecules endpoint failed with error: {e}")
        return False
    
    # Test predict endpoint
    print("\n3. Testing predict endpoint...")
    try:
        response = requests.post(
            f"{base_url}/predict",
            json={"smiles": sample_smiles}
        )
        if response.status_code == 200:
            prediction = response.json()
            print("✅ Prediction endpoint passed")
            print(f"Prediction for {sample_name}:")
            print(f"  Solubility: {prediction['predicted_solubility']}")
            print(f"  Level: {prediction['solubility_level']}")
        else:
            print(f"❌ Prediction endpoint failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction endpoint failed with error: {e}")
        return False
    
    # Test batch predict endpoint
    print("\n4. Testing batch predict endpoint...")
    try:
        # Get multiple sample SMILES
        batch_smiles = [sample["smiles"] for sample in data["samples"][:3]]
        response = requests.post(
            f"{base_url}/batch-predict",
            json={"smiles_list": batch_smiles}
        )
        if response.status_code == 200:
            batch_results = response.json()
            if "predictions" in batch_results and len(batch_results["predictions"]) > 0:
                print(f"✅ Batch prediction endpoint passed. Processed {len(batch_results['predictions'])} molecules")
            else:
                print("❌ Batch prediction endpoint failed: No predictions found")
                return False
        else:
            print(f"❌ Batch prediction endpoint failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Batch prediction endpoint failed with error: {e}")
        return False
    
    # Test visualize molecule endpoint
    print("\n5. Testing visualize molecule endpoint...")
    try:
        response = requests.post(
            f"{base_url}/visualize-molecule",
            data={"smiles": sample_smiles}
        )
        if response.status_code == 200:
            visualization = response.json()
            if "image" in visualization:
                print("✅ Visualization endpoint passed")
                # Optionally save the image
                try:
                    image_data = base64.b64decode(visualization["image"])
                    image = Image.open(BytesIO(image_data))
                    image_path = f"{sample_name.lower().replace(' ', '_')}_visualization.png"
                    image.save(image_path)
                    print(f"Visualization saved to {image_path}")
                except Exception as e:
                    print(f"Warning: Could not save visualization: {e}")
            else:
                print("❌ Visualization endpoint failed: No image in response")
                return False
        else:
            print(f"❌ Visualization endpoint failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Visualization endpoint failed with error: {e}")
        return False
    
    # Test predict with visualization endpoint
    print("\n6. Testing predict with visualization endpoint...")
    try:
        response = requests.post(
            f"{base_url}/predict-with-visualization",
            data={"smiles": sample_smiles}
        )
        if response.status_code == 200:
            prediction_with_viz = response.json()
            if "image" in prediction_with_viz and "predicted_solubility" in prediction_with_viz:
                print("✅ Predict with visualization endpoint passed")
            else:
                print("❌ Predict with visualization endpoint failed: Incomplete response")
                return False
        else:
            print(f"❌ Predict with visualization endpoint failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Predict with visualization endpoint failed with error: {e}")
        return False
    
    # Test validate SMILES endpoint
    print("\n7. Testing validate SMILES endpoint...")
    try:
        response = requests.post(
            f"{base_url}/validate-smiles",
            data={"smiles": sample_smiles}
        )
        if response.status_code == 200:
            validation = response.json()
            if "valid" in validation and validation["valid"]:
                print("✅ Validate SMILES endpoint passed")
            else:
                print("❌ Validate SMILES endpoint failed: SMILES not validated")
                return False
        else:
            print(f"❌ Validate SMILES endpoint failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Validate SMILES endpoint failed with error: {e}")
        return False
    
    # Test model info endpoint
    print("\n8. Testing model info endpoint...")
    try:
        response = requests.get(f"{base_url}/model-info")
        if response.status_code == 200:
            model_info = response.json()
            if "model_info" in model_info:
                print("✅ Model info endpoint passed")
                print(f"Model info: {json.dumps(model_info['model_info'], indent=2)}")
            else:
                print("❌ Model info endpoint failed: No model info in response")
                return False
        else:
            print(f"❌ Model info endpoint failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Model info endpoint failed with error: {e}")
        return False
    
    print("\n✅ All tests passed successfully!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Molecular Solubility Prediction API")
    parser.add_argument("--url", type=str, default="http://localhost:8000", 
                        help="Base URL of the API (default: http://localhost:8000)")
    parser.add_argument("--wait", type=int, default=0,
                        help="Wait for the specified number of seconds before starting tests")
    args = parser.parse_args()
    
    # Wait if requested
    if args.wait > 0:
        print(f"Waiting {args.wait} seconds before starting tests...")
        time.sleep(args.wait)
    
    # Run the tests
    success = test_api(args.url)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)