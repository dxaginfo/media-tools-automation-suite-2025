# SceneValidator

A tool for validating scene composition and continuity in media projects.

## Overview

SceneValidator analyzes video content to identify issues with composition, lighting consistency, and continuity across scenes. It leverages computer vision and AI technologies to provide actionable feedback for improving video quality.

## Features

- **Composition Analysis**: Evaluates adherence to rule of thirds, framing, balance, and visual flow
- **Lighting Consistency**: Detects brightness, color temperature, and light direction changes between scenes
- **Continuity Tracking**: Identifies missing objects or significant position changes across scenes
- **Scene Break Detection**: Automatically identifies scene transitions
- **Comprehensive Reporting**: Generates detailed reports with issue severity and improvement suggestions

## Implementation Details

### Core Technologies

- Python for core functionality
- OpenCV for image processing
- Google Cloud Vision API for object detection
- Gemini API for advanced composition analysis
- Firebase for report storage (optional)

### Integration Points

- Input: Video files in standard formats (MP4, MOV, AVI)
- Output: JSON reports, optionally with HTML visualization
- Firebase integration for report storage and sharing
- Google Cloud Storage for handling large media files

## Requirements

- Python 3.8+
- OpenCV 4.5+
- Optional: Google Cloud credentials for Vision API
- Optional: Gemini API key for advanced analysis
- Optional: Firebase credentials for cloud storage

## Installation

```bash
# Install required packages
pip install -r requirements.txt

# Set up Google Cloud credentials (optional)
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"

# Set up Gemini API key (optional)
export GEMINI_API_KEY="your_api_key"
```

## Usage

### Basic Usage

```bash
python scene_validator.py /path/to/video.mp4
```

### With Configuration

```bash
python scene_validator.py /path/to/video.mp4 --config config.json --output ./reports
```

## Configuration

See the [configuration template](./config/config_template.json) for all available options.

## Integration with Other Tools

SceneValidator works well with:

- **TimelineAssembler**: For correcting identified issues in timeline sequences
- **ContinuityTracker**: For deeper continuity analysis between scenes
- **StoryboardGen**: For comparing actual footage against intended storyboard compositions