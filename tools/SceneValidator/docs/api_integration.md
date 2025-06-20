# SceneValidator API Integration Guide

This document provides detailed information on how to integrate SceneValidator with Google APIs and other services.

## Table of Contents

1. [Gemini API Integration](#gemini-api-integration)
2. [Google Cloud Vision API Integration](#google-cloud-vision-api-integration)
3. [Firebase Integration](#firebase-integration)
4. [REST API Reference](#rest-api-reference)
5. [Webhook Integration](#webhook-integration)
6. [Integration with Other Tools](#integration-with-other-tools)

## Gemini API Integration

SceneValidator uses Google's Gemini API for advanced scene analysis and recommendations.

### Setup

1. Create a Google Cloud project at [https://console.cloud.google.com/](https://console.cloud.google.com/)
2. Enable the Gemini API for your project
3. Create an API key from the credentials page
4. Add the API key to your SceneValidator configuration

```json
{
  "api_credentials": {
    "gemini_api_key": "YOUR_GEMINI_API_KEY"
  }
}
```

### Usage in SceneValidator

SceneValidator uses Gemini for:

1. **Scene Composition Analysis**: Evaluates adherence to cinematography principles
2. **Anomaly Detection**: Identifies unusual elements or inconsistencies in scenes
3. **Recommendation Generation**: Creates suggestions for fixing detected issues

```python
import google.generativeai as genai

# Initialize the Gemini model
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro-vision')

# Analyze a frame
def analyze_composition(frame, context):
    # Convert frame to appropriate format
    image_parts = [{"mime_type": "image/jpeg", "data": frame_to_bytes(frame)}]
    
    # Create prompt with context
    prompt = f"""
    Analyze this video frame for composition issues:
    - Check rule of thirds adherence
    - Evaluate framing and headroom
    - Check leading lines and visual flow
    - Assess balance and symmetry

    Context: {context}
    """
    
    # Get analysis from Gemini
    response = model.generate_content([prompt, *image_parts])
    
    # Process and return results
    return parse_gemini_response(response.text)
```

## Google Cloud Vision API Integration

SceneValidator uses the Vision API for object detection and tracking across scenes.

### Setup

1. Enable the Vision API in your Google Cloud project
2. Set up authentication using service account credentials
3. Configure the client in SceneValidator

```python
from google.cloud import vision

# Create a client
client = vision.ImageAnnotatorClient()

# Detect objects in a frame
def detect_objects(frame):
    # Convert frame to Vision API format
    image = vision.Image(content=frame_to_bytes(frame))
    
    # Perform object detection
    response = client.object_localization(image=image)
    objects = response.localized_object_annotations
    
    # Process results
    detected_objects = []
    for obj in objects:
        detected_objects.append({
            'name': obj.name,
            'confidence': obj.score,
            'bounding_box': extract_bounding_box(obj.bounding_poly)
        })
    
    return detected_objects
```

## Firebase Integration

SceneValidator uses Firebase for storing validation reports and user authentication.

### Setup

1. Create a Firebase project (can be linked to your existing Google Cloud project)
2. Set up Firestore database
3. Create a service account and download credentials JSON
4. Configure SceneValidator with your Firebase credentials

```python
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase
cred = credentials.Certificate('path/to/firebase_credentials.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Save a validation report
def save_report(project_id, report_data):
    reports_ref = db.collection('validation_reports')
    reports_ref.add({
        'project_id': project_id,
        'report': report_data,
        'created_at': firestore.SERVER_TIMESTAMP
    })
```

## REST API Reference

SceneValidator provides a REST API for integration with other tools and services.

### Authentication

All API requests require authentication using an API key:

```
Authorization: Bearer YOUR_API_KEY
```

### Endpoints

#### `POST /api/v1/validate`

Submit a video for validation.

**Request:**

```json
{
  "video_url": "https://example.com/video.mp4",
  "settings": {
    "sensitivity": 0.7,
    "check_composition": true,
    "check_lighting": true,
    "check_continuity": true
  },
  "callback_url": "https://your-service.com/webhook"
}
```

**Response:**

```json
{
  "job_id": "val_1234567890",
  "status": "processing",
  "estimated_completion_time": "2025-06-20T18:30:00Z"
}
```

#### `GET /api/v1/reports/{job_id}`

Get validation results for a job.

**Response:**

```json
{
  "job_id": "val_1234567890",
  "status": "completed",
  "video_info": {
    "duration": 125.5,
    "resolution": "1920x1080",
    "fps": 24
  },
  "issues": [
    {
      "issue_type": "composition",
      "timestamp": 12.5,
      "description": "Rule of thirds violation",
      "severity": 0.6,
      "frame_number": 300,
      "suggestions": [
        "Reframe shot to place subject at intersection of thirds grid"
      ]
    }
  ],
  "summary": {
    "total_issues": 7,
    "high_severity": 2,
    "medium_severity": 3,
    "low_severity": 2
  },
  "report_urls": {
    "html": "https://scene-validator.example.com/reports/val_1234567890.html",
    "json": "https://scene-validator.example.com/reports/val_1234567890.json"
  }
}
```

## Webhook Integration

SceneValidator can send webhook notifications when validation is complete.

### Payload Format

```json
{
  "event": "validation.completed",
  "job_id": "val_1234567890",
  "project_id": "project_9876543210",
  "status": "completed",
  "summary": {
    "total_issues": 7,
    "high_severity": 2,
    "medium_severity": 3,
    "low_severity": 2
  },
  "report_urls": {
    "html": "https://scene-validator.example.com/reports/val_1234567890.html",
    "json": "https://scene-validator.example.com/reports/val_1234567890.json"
  },
  "timestamp": "2025-06-20T18:30:00Z"
}
```

## Integration with Other Tools

SceneValidator is designed to work with other tools in the Media Automation Suite.

### TimelineAssembler Integration

```python
# Example of integrating with TimelineAssembler
import requests

def send_to_timeline_assembler(validation_report, timeline_api_url):
    # Extract scene breaks and issues from validation report
    scene_data = extract_scene_data(validation_report)
    
    # Send to TimelineAssembler API
    response = requests.post(
        f"{timeline_api_url}/scenes",
        json=scene_data,
        headers={"Authorization": f"Bearer {api_key}"}
    )
    
    return response.json()
```

### ContinuityTracker Integration

SceneValidator can pass detected continuity issues to ContinuityTracker for more detailed analysis:

```python
def update_continuity_tracker(continuity_issues, tracker_api_url):
    # Prepare data for ContinuityTracker
    continuity_data = {
        "project_id": project_id,
        "issues": continuity_issues,
        "detected_objects": detected_objects
    }
    
    # Send to ContinuityTracker API
    response = requests.post(
        f"{tracker_api_url}/issues/batch",
        json=continuity_data,
        headers={"Authorization": f"Bearer {api_key}"}
    )
    
    return response.json()
```

### FormatNormalizer Integration

When scene format issues are detected, SceneValidator can request normalization:

```python
def request_format_normalization(format_issues, normalizer_api_url):
    # Prepare normalization request
    normalization_data = {
        "video_url": video_url,
        "issues": format_issues,
        "target_format": {
            "resolution": "1920x1080",
            "frame_rate": 24,
            "color_profile": "Rec.709"
        }
    }
    
    # Send to FormatNormalizer API
    response = requests.post(
        f"{normalizer_api_url}/normalize",
        json=normalization_data,
        headers={"Authorization": f"Bearer {api_key}"}
    )
    
    return response.json()
```