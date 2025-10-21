#!/usr/bin/env python3
"""
Test script for the AI Career Guidance API
Demonstrates the system's capabilities with sample requests
"""

import requests
import json
import time

# API Base URL
BASE_URL = "http://localhost:5000"

def test_health():
    """Test the health endpoint"""
    print("🔍 Testing Health Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   Health Status: {data['status']}")
            print(f"   Model Loaded: {data['model_loaded']}")
            print(f"   Encoder Loaded: {data['encoder_loaded']}")
            print(f"   Careers Available: {data['careers_available']}")
        else:
            print(f"   Health check failed: {response.status_code}")
    except Exception as e:
        print(f"  Health check error: {e}")

def test_careers():
    """Test the careers endpoint"""
    print("\n Testing Careers Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/careers")
        if response.status_code == 200:
            data = response.json()
            print(f" Found {data['total']} careers")
            for career in data['careers'][:3]:  # Show first 3
                print(f"   - {career['name']}: {career['salary']}")
        else:
            print(f" Careers endpoint failed: {response.status_code}")
    except Exception as e:
        print(f" Careers endpoint error: {e}")

def test_prediction():
    """Test the prediction endpoint with sample data"""
    print("\n Testing Prediction Endpoint...")
    
    # Sample test cases
    test_cases = [
        {
            "name": "Alice Johnson",
            "cgpa": 8.7,
            "skills": ["Python", "Machine Learning", "Statistics"],
            "interests": ["Artificial Intelligence", "Data Science"],
            "description": "AI/ML Enthusiast"
        },
        {
            "name": "Bob Smith",
            "cgpa": 7.5,
            "skills": ["JavaScript", "React", "HTML", "CSS"],
            "interests": ["Web Development", "Entrepreneurship"],
            "description": "Web Developer"
        },
        {
            "name": "Carol Davis",
            "cgpa": 8.9,
            "skills": ["Python", "TensorFlow", "Research"],
            "interests": ["Research", "Artificial Intelligence"],
            "description": "Research Scientist"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n   Test Case {i}: {test_case['description']}")
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json=test_case,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   Predicted Career: {data['career']}")
                print(f"   Match Percentage: {data['match_percentage']}%")
                print(f"   Salary: {data['info']['salary']}")
                print(f"   Companies: {len(data['info']['companies'])} options")
                print(f"   Courses: {len(data['info']['courses'])} recommendations")
                print(f"   Roadmap: {len(data['info']['roadmap'])} steps")
                
                if data['alternatives']:
                    print(f"   Alternatives: {', '.join([alt['career'] for alt in data['alternatives'][:2]])}")
            else:
                print(f"   Prediction failed: {response.status_code}")
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"   Prediction error: {e}")
        
        time.sleep(1)  # Small delay between requests

def test_career_details():
    """Test getting specific career details"""
    print("\n🔍 Testing Career Details Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/career/Data Scientist")
        if response.status_code == 200:
            data = response.json()
            career = data['career']
            info = data['info']
            print(f"✅ Career: {career}")
            print(f"   💰 Salary: {info['salary']}")
            print(f"   🏢 Companies: {len(info['companies'])} options")
            print(f"   📚 Courses: {len(info['courses'])} recommendations")
            print(f"   🛣️  Roadmap: {len(info['roadmap'])} steps")
        else:
            print(f"❌ Career details failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Career details error: {e}")

def main():
    """Run all tests"""
    print("🚀 AI Career Guidance System - API Test Suite")
    print("=" * 50)
    
    # Run all tests
    test_health()
    test_careers()
    test_prediction()
    test_career_details()
    
    print("\n" + "=" * 50)
    print(" All tests completed!")
    print("\n Tips:")
    print("   - Make sure the backend server is running (python app.py)")
    print("   - Check that the model files exist in the models/ directory")
    print("   - The system provides personalized career recommendations")
    print("   - Each prediction includes match percentage, salary, companies, courses, and roadmap")

if __name__ == "__main__":
    main()
