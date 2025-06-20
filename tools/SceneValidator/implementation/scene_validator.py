#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SceneValidator: A tool for validating scene composition and continuity in media projects.

This module provides the core functionality for analyzing video scenes to identify
composition issues, lighting inconsistencies, and continuity problems.
"""

import os
import json
import logging
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Import optional dependencies - will be checked at runtime
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from google.cloud import vision
    VISION_API_AVAILABLE = True
except ImportError:
    VISION_API_AVAILABLE = False

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SceneValidator')


class SceneValidator:
    """
    Main class for validating scene composition and continuity in media projects.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the SceneValidator with configuration.
        
        Args:
            config_path: Path to configuration JSON file
        """
        self.config = self._load_config(config_path)
        self.frames = []
        self.scene_breaks = []
        self.issues = []
        self.report = {}
        
        # Initialize APIs based on configuration
        self._init_apis()
        
        logger.info("SceneValidator initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from a JSON file or use defaults.
        
        Args:
            config_path: Path to configuration JSON file
            
        Returns:
            Dict containing configuration
        """
        default_config = {
            "sensitivity": 0.7,
            "check_composition": True,
            "check_lighting": True,
            "check_continuity": True,
            "api_credentials": {},
            "output_dir": "./output",
            "temp_dir": "/tmp/scenevalidator"
        }
        
        if not config_path:
            logger.warning("No config file provided, using defaults")
            return default_config
        
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Update default config with user-provided values
                default_config.update(user_config)
                logger.info(f"Loaded configuration from {config_path}")
                return default_config
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Failed to load config from {config_path}: {str(e)}")
            logger.warning("Using default configuration")
            return default_config
    
    def _init_apis(self):
        """Initialize API clients based on configuration and availability."""
        # Initialize Gemini API if available and configured
        self.gemini_model = None
        if GEMINI_AVAILABLE and 'gemini_api_key' in self.config['api_credentials']:
            try:
                api_key = self.config['api_credentials']['gemini_api_key']
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel('gemini-pro-vision')
                logger.info("Gemini API initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini API: {str(e)}")
        
        # Initialize Vision API if available and configured
        self.vision_client = None
        if VISION_API_AVAILABLE:
            try:
                self.vision_client = vision.ImageAnnotatorClient()
                logger.info("Vision API initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Vision API: {str(e)}")
        
        # Initialize Firebase if available and configured
        self.db = None
        if FIREBASE_AVAILABLE and 'firebase_credentials' in self.config['api_credentials']:
            try:
                cred_path = self.config['api_credentials']['firebase_credentials']
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
                self.db = firestore.client()
                logger.info("Firebase initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Firebase: {str(e)}")
    
    def validate_video(self, video_path: str) -> Dict:
        """
        Validate a video file and generate a report of issues.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dict containing validation report
        """
        if not CV2_AVAILABLE:
            logger.error("OpenCV (cv2) is required for video processing")
            return {"error": "OpenCV is not available"}
            
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return {"error": "Video file not found"}
        
        logger.info(f"Starting validation of {video_path}")
        
        # Extract frames
        self._extract_frames(video_path)
        
        # Detect scene breaks
        self._detect_scene_breaks()
        
        # Analyze for issues
        if self.config['check_composition']:
            self._analyze_composition()
        
        if self.config['check_lighting']:
            self._analyze_lighting()
        
        if self.config['check_continuity']:
            self._analyze_continuity()
        
        # Generate report
        self._generate_report(video_path)
        
        # Save report
        self._save_report()
        
        logger.info(f"Validation completed with {len(self.issues)} issues found")
        
        return self.report
    
    def _extract_frames(self, video_path: str, sample_rate: int = 24):
        """
        Extract frames from the video at a specified sample rate.
        
        Args:
            video_path: Path to the video file
            sample_rate: Number of frames to extract per second
        """
        logger.info(f"Extracting frames from {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Failed to open video file")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        # Store video info
        self.video_info = {
            "path": video_path,
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
        
        # Calculate frame extraction interval
        interval = int(fps / sample_rate)
        if interval < 1:
            interval = 1
        
        # Extract frames
        self.frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % interval == 0:
                frame_info = {
                    "index": frame_idx,
                    "timestamp": frame_idx / fps,
                    "data": frame
                }
                self.frames.append(frame_info)
            
            frame_idx += 1
        
        cap.release()
        logger.info(f"Extracted {len(self.frames)} frames from video")
    
    def _detect_scene_breaks(self, threshold: float = 0.35):
        """
        Detect scene breaks by analyzing frame differences.
        
        Args:
            threshold: Difference threshold for scene break detection
        """
        logger.info("Detecting scene breaks")
        
        self.scene_breaks = []
        
        if len(self.frames) < 2:
            logger.warning("Not enough frames to detect scene breaks")
            return
        
        # Calculate differences between consecutive frames
        for i in range(1, len(self.frames)):
            prev_frame = self.frames[i-1]["data"]
            curr_frame = self.frames[i]["data"]
            
            # Convert frames to grayscale
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate histograms
            prev_hist = cv2.calcHist([prev_gray], [0], None, [64], [0, 256])
            curr_hist = cv2.calcHist([curr_gray], [0], None, [64], [0, 256])
            
            # Normalize histograms
            cv2.normalize(prev_hist, prev_hist, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(curr_hist, curr_hist, 0, 1, cv2.NORM_MINMAX)
            
            # Calculate histogram difference
            diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA)
            
            # If difference exceeds threshold, mark as scene break
            if diff > threshold:
                scene_break = {
                    "frame_index": i,
                    "timestamp": self.frames[i]["timestamp"],
                    "difference": diff
                }
                self.scene_breaks.append(scene_break)
        
        logger.info(f"Detected {len(self.scene_breaks)} scene breaks")
    
    def _analyze_composition(self):
        """Analyze frame composition for issues."""
        logger.info("Analyzing composition")
        
        # Check if Gemini is available for advanced analysis
        if self.gemini_model:
            self._analyze_composition_with_gemini()
        else:
            self._analyze_composition_basic()
    
    def _analyze_composition_basic(self):
        """Basic composition analysis using OpenCV."""
        for frame_info in self.frames:
            frame = frame_info["data"]
            frame_idx = frame_info["index"]
            timestamp = frame_info["timestamp"]
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Check rule of thirds - look for high edge density at third points
            height, width = gray.shape
            third_h = height // 3
            third_w = width // 3
            
            # Get regions at rule of thirds intersections
            regions = [
                gray[third_h-10:third_h+10, third_w-10:third_w+10],
                gray[third_h-10:third_h+10, 2*third_w-10:2*third_w+10],
                gray[2*third_h-10:2*third_h+10, third_w-10:third_w+10],
                gray[2*third_h-10:2*third_h+10, 2*third_w-10:2*third_w+10]
            ]
            
            # Apply edge detection
            edges = cv2.Canny(gray, 100, 200)
            
            # Check for edges at rule of thirds points
            roi_edges = [
                edges[third_h-10:third_h+10, third_w-10:third_w+10],
                edges[third_h-10:third_h+10, 2*third_w-10:2*third_w+10],
                edges[2*third_h-10:2*third_h+10, third_w-10:third_w+10],
                edges[2*third_h-10:2*third_h+10, 2*third_w-10:2*third_w+10]
            ]
            
            # Calculate edge density in ROIs
            edge_densities = [np.sum(roi > 0) / roi.size for roi in roi_edges]
            max_density = max(edge_densities)
            
            # If no significant edges at rule of thirds points, flag as issue
            if max_density < 0.1:  # Threshold for significant edges
                issue = {
                    "issue_type": "composition",
                    "subtype": "rule_of_thirds",
                    "description": "Low rule of thirds utilization",
                    "frame_index": frame_idx,
                    "timestamp": timestamp,
                    "severity": 0.5 - max_density,  # Severity based on edge density
                    "suggestions": [
                        "Consider placing key elements at rule of thirds intersections"
                    ]
                }
                self.issues.append(issue)
    
    def _analyze_composition_with_gemini(self):
        """
        Advanced composition analysis using Gemini API.
        This function selects key frames (including scene breaks) for analysis.
        """
        # Select frames to analyze (scene breaks and regular samples)
        scene_break_indices = [sb["frame_index"] for sb in self.scene_breaks]
        
        # Add regular samples if we don't have enough scene breaks
        analyze_indices = set(scene_break_indices)
        if len(analyze_indices) < 10:
            # Add more frames spaced evenly throughout the video
            total_frames = len(self.frames)
            samples_needed = 10 - len(analyze_indices)
            step = total_frames // (samples_needed + 1)
            
            for i in range(1, samples_needed + 1):
                analyze_indices.add(i * step)
        
        # Convert to list and sort
        analyze_indices = sorted(list(analyze_indices))
        
        # Analyze selected frames
        for idx in analyze_indices:
            if idx >= len(self.frames):
                continue
                
            frame_info = self.frames[idx]
            frame = frame_info["data"]
            timestamp = frame_info["timestamp"]
            
            # Convert frame to format accepted by Gemini
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                logger.warning(f"Failed to encode frame {idx} for Gemini analysis")
                continue
                
            image_bytes = buffer.tobytes()
            
            # Create prompt for Gemini
            prompt = """
            Analyze this video frame for composition issues:
            1. Check rule of thirds adherence
            2. Evaluate framing and headroom
            3. Check leading lines and visual flow
            4. Assess balance and symmetry
            5. Evaluate color composition
            
            Respond with a JSON object containing the following fields:
            - has_issues (boolean): Whether the frame has composition issues
            - issue_type (string): Type of composition issue (if any)
            - description (string): Detailed description of the issue
            - severity (float 0-1): How severe the issue is
            - suggestions (array of strings): Suggestions to fix the issue
            """
            
            # Call Gemini API
            try:
                image_parts = [{"mime_type": "image/jpeg", "data": image_bytes}]
                response = self.gemini_model.generate_content([prompt, *image_parts])
                
                # Parse response
                try:
                    # Extract JSON from response
                    response_text = response.text
                    # Find JSON content (anything between { and })
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}')
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = response_text[json_start:json_end+1]
                        analysis = json.loads(json_str)
                        
                        # Add to issues if has_issues is True
                        if analysis.get('has_issues', False):
                            issue = {
                                "issue_type": "composition",
                                "subtype": analysis.get('issue_type', 'unknown'),
                                "description": analysis.get('description', ''),
                                "frame_index": idx,
                                "timestamp": timestamp,
                                "severity": float(analysis.get('severity', 0.5)),
                                "suggestions": analysis.get('suggestions', [])
                            }
                            self.issues.append(issue)
                    else:
                        logger.warning(f"No valid JSON found in Gemini response for frame {idx}")
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Failed to parse Gemini response for frame {idx}: {str(e)}")
            except Exception as e:
                logger.error(f"Error calling Gemini API for frame {idx}: {str(e)}")
    
    def _analyze_lighting(self):
        """Analyze lighting consistency across scenes."""
        logger.info("Analyzing lighting consistency")
        
        if len(self.scene_breaks) < 1:
            logger.warning("No scene breaks detected for lighting analysis")
            return
            
        # Get representative frames for each scene
        scene_frames = []
        
        # Add first frame
        scene_frames.append(self.frames[0])
        
        # Add frame after each scene break
        for scene_break in self.scene_breaks:
            idx = scene_break["frame_index"]
            if idx + 1 < len(self.frames):
                scene_frames.append(self.frames[idx + 1])
        
        # Analyze lighting characteristics for each scene
        scene_lighting = []
        for frame_info in scene_frames:
            frame = frame_info["data"]
            frame_idx = frame_info["index"]
            timestamp = frame_info["timestamp"]
            
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Calculate average brightness (V in HSV)
            brightness = np.mean(hsv[:,:,2])
            
            # Calculate color temperature (approximation based on B-R ratio)
            bgr_means = np.mean(frame, axis=(0,1))
            b_r_ratio = bgr_means[0] / max(bgr_means[2], 1)  # Avoid division by zero
            
            # Higher ratio = cooler temperature, lower = warmer
            if b_r_ratio > 1.1:
                temperature = "cool"
            elif b_r_ratio < 0.9:
                temperature = "warm"
            else:
                temperature = "neutral"
            
            # Detect main light direction (simplified)
            # This is a basic approach - a more advanced method would use shadow detection
            edges = cv2.Canny(frame, 100, 200)
            gradient_y = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)
            gradient_x = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
            
            direction_x = np.mean(gradient_x)
            direction_y = np.mean(gradient_y)
            
            if abs(direction_x) > abs(direction_y):
                if direction_x > 0:
                    light_direction = "right"
                else:
                    light_direction = "left"
            else:
                if direction_y > 0:
                    light_direction = "bottom"
                else:
                    light_direction = "top"
            
            # Store lighting info
            lighting_info = {
                "frame_index": frame_idx,
                "timestamp": timestamp,
                "brightness": brightness,
                "temperature": temperature,
                "light_direction": light_direction
            }
            scene_lighting.append(lighting_info)
        
        # Compare lighting across scenes
        for i in range(1, len(scene_lighting)):
            prev_lighting = scene_lighting[i-1]
            curr_lighting = scene_lighting[i]
            
            # Check for brightness inconsistency
            brightness_diff = abs(prev_lighting["brightness"] - curr_lighting["brightness"])
            if brightness_diff > 50:  # Threshold for significant brightness change
                issue = {
                    "issue_type": "lighting",
                    "subtype": "brightness",
                    "description": f"Significant brightness change between scenes ({brightness_diff:.1f} units)",
                    "frame_index": curr_lighting["frame_index"],
                    "timestamp": curr_lighting["timestamp"],
                    "severity": min(brightness_diff / 100, 1.0),  # Normalize to 0-1
                    "prev_timestamp": prev_lighting["timestamp"],
                    "suggestions": [
                        "Adjust brightness in post-processing",
                        "Re-light scene for consistency"
                    ]
                }
                self.issues.append(issue)
            
            # Check for color temperature inconsistency
            if prev_lighting["temperature"] != curr_lighting["temperature"]:
                issue = {
                    "issue_type": "lighting",
                    "subtype": "color_temperature",
                    "description": f"Color temperature change from {prev_lighting['temperature']} to {curr_lighting['temperature']}",
                    "frame_index": curr_lighting["frame_index"],
                    "timestamp": curr_lighting["timestamp"],
                    "severity": 0.7,  # Fixed severity for temperature changes
                    "prev_timestamp": prev_lighting["timestamp"],
                    "suggestions": [
                        "Adjust white balance in post-processing",
                        "Use consistent lighting temperature"
                    ]
                }
                self.issues.append(issue)
            
            # Check for light direction inconsistency
            if prev_lighting["light_direction"] != curr_lighting["light_direction"]:
                issue = {
                    "issue_type": "lighting",
                    "subtype": "direction",
                    "description": f"Light direction change from {prev_lighting['light_direction']} to {curr_lighting['light_direction']}",
                    "frame_index": curr_lighting["frame_index"],
                    "timestamp": curr_lighting["timestamp"],
                    "severity": 0.6,  # Fixed severity for direction changes
                    "prev_timestamp": prev_lighting["timestamp"],
                    "suggestions": [
                        "Ensure consistent light placement across scenes",
                        "Consider re-shooting with matching light direction"
                    ]
                }
                self.issues.append(issue)
    
    def _analyze_continuity(self):
        """Analyze continuity issues across scenes."""
        logger.info("Analyzing continuity")
        
        # Skip if no Vision API client or no scene breaks
        if not self.vision_client or len(self.scene_breaks) < 1:
            if not self.vision_client:
                logger.warning("Vision API client not available for continuity analysis")
            else:
                logger.warning("No scene breaks detected for continuity analysis")
            return
        
        # Get representative frames for each scene
        scene_frames = []
        
        # Add first frame
        scene_frames.append(self.frames[0])
        
        # Add frame after each scene break
        for scene_break in self.scene_breaks:
            idx = scene_break["frame_index"]
            if idx + 1 < len(self.frames):
                scene_frames.append(self.frames[idx + 1])
        
        # Detect objects in each scene
        scene_objects = []
        for frame_info in scene_frames:
            frame = frame_info["data"]
            frame_idx = frame_info["index"]
            timestamp = frame_info["timestamp"]
            
            # Convert frame to Vision API format
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                logger.warning(f"Failed to encode frame {frame_idx} for Vision API")
                continue
                
            image_bytes = buffer.tobytes()
            image = vision.Image(content=image_bytes)
            
            # Perform object detection
            try:
                response = self.vision_client.object_localization(image=image)
                objects = response.localized_object_annotations
                
                # Store detected objects
                detected_objects = []
                for obj in objects:
                    # Extract vertices
                    vertices = []
                    for vertex in obj.bounding_poly.normalized_vertices:
                        vertices.append((vertex.x, vertex.y))
                    
                    # Calculate center point
                    center_x = sum(v[0] for v in vertices) / len(vertices)
                    center_y = sum(v[1] for v in vertices) / len(vertices)
                    
                    # Store object info
                    object_info = {
                        "name": obj.name,
                        "score": obj.score,
                        "vertices": vertices,
                        "center": (center_x, center_y)
                    }
                    detected_objects.append(object_info)
                
                scene_objects.append({
                    "frame_index": frame_idx,
                    "timestamp": timestamp,
                    "objects": detected_objects
                })
            except Exception as e:
                logger.error(f"Error in Vision API object detection for frame {frame_idx}: {str(e)}")
        
        # Compare objects across scenes to detect continuity issues
        for i in range(1, len(scene_objects)):
            prev_scene = scene_objects[i-1]
            curr_scene = scene_objects[i]
            
            # Look for missing objects (were in previous scene but not in current)
            prev_object_names = {obj["name"] for obj in prev_scene["objects"]}
            curr_object_names = {obj["name"] for obj in curr_scene["objects"]}
            
            missing_objects = prev_object_names - curr_object_names
            
            # Check if important objects disappeared
            for obj_name in missing_objects:
                # Only flag if the object had high confidence
                prev_obj = next((o for o in prev_scene["objects"] if o["name"] == obj_name), None)
                if prev_obj and prev_obj["score"] > 0.7:
                    issue = {
                        "issue_type": "continuity",
                        "subtype": "missing_object",
                        "description": f"Object '{obj_name}' present in previous scene is missing",
                        "frame_index": curr_scene["frame_index"],
                        "timestamp": curr_scene["timestamp"],
                        "severity": prev_obj["score"] - 0.5,  # Higher confidence = higher severity
                        "prev_timestamp": prev_scene["timestamp"],
                        "object": obj_name,
                        "suggestions": [
                            f"Ensure {obj_name} is consistently present across related scenes",
                            "Check for continuity in object placement"
                        ]
                    }
                    self.issues.append(issue)
            
            # Look for objects that appear in both scenes but changed position significantly
            common_objects = prev_object_names.intersection(curr_object_names)
            for obj_name in common_objects:
                prev_obj = next((o for o in prev_scene["objects"] if o["name"] == obj_name), None)
                curr_obj = next((o for o in curr_scene["objects"] if o["name"] == obj_name), None)
                
                if prev_obj and curr_obj:
                    # Calculate position change
                    prev_center = prev_obj["center"]
                    curr_center = curr_obj["center"]
                    
                    distance = ((prev_center[0] - curr_center[0])**2 + 
                               (prev_center[1] - curr_center[1])**2)**0.5
                    
                    # Flag if position changed significantly
                    if distance > 0.3:  # Normalized distance threshold
                        issue = {
                            "issue_type": "continuity",
                            "subtype": "position_change",
                            "description": f"Object '{obj_name}' changed position significantly between scenes",
                            "frame_index": curr_scene["frame_index"],
                            "timestamp": curr_scene["timestamp"],
                            "severity": min(distance, 1.0),
                            "prev_timestamp": prev_scene["timestamp"],
                            "object": obj_name,
                            "suggestions": [
                                f"Ensure consistent positioning of {obj_name} across scenes",
                                "Use continuity markers during filming"
                            ]
                        }
                        self.issues.append(issue)
    
    def _generate_report(self, video_path: str):
        """
        Generate a comprehensive report of all detected issues.
        
        Args:
            video_path: Path to the original video file
        """
        logger.info("Generating validation report")
        
        # Count issues by type and severity
        issue_counts = {"composition": 0, "lighting": 0, "continuity": 0}
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        
        for issue in self.issues:
            # Count by type
            issue_type = issue["issue_type"]
            if issue_type in issue_counts:
                issue_counts[issue_type] += 1
            
            # Count by severity
            severity = issue["severity"]
            if severity >= 0.7:
                severity_counts["high"] += 1
            elif severity >= 0.4:
                severity_counts["medium"] += 1
            else:
                severity_counts["low"] += 1
        
        # Generate report structure
        self.report = {
            "video_path": video_path,
            "video_info": self.video_info,
            "analysis_time": datetime.now().isoformat(),
            "scene_breaks": len(self.scene_breaks),
            "total_issues": len(self.issues),
            "issue_summary": {
                "by_type": issue_counts,
                "by_severity": severity_counts
            },
            "issues": self.issues
        }
    
    def _save_report(self):
        """Save the validation report to file and Firebase if configured."""
        if not self.report:
            logger.warning("No report to save")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config["output_dir"], exist_ok=True)
        
        # Generate filename based on video path
        video_filename = os.path.basename(self.report["video_path"])
        base_name = os.path.splitext(video_filename)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"{base_name}_validation_{timestamp}.json"
        report_path = os.path.join(self.config["output_dir"], report_filename)
        
        # Save to file
        try:
            # Create a copy of the report without frame data to reduce size
            save_report = self.report.copy()
            
            # Save to JSON file
            with open(report_path, 'w') as f:
                json.dump(save_report, f, indent=2)
            
            logger.info(f"Report saved to {report_path}")
            
            # Also save to Firebase if available
            if self.db:
                try:
                    reports_ref = self.db.collection('validation_reports')
                    reports_ref.add({
                        'report': save_report,
                        'created_at': firestore.SERVER_TIMESTAMP
                    })
                    logger.info("Report saved to Firebase")
                except Exception as e:
                    logger.error(f"Failed to save report to Firebase: {str(e)}")
        
        except Exception as e:
            logger.error(f"Failed to save report: {str(e)}")


def main():
    """Command line interface for SceneValidator."""
    parser = argparse.ArgumentParser(description="SceneValidator: Media scene validation tool")
    parser.add_argument("video_path", help="Path to the video file to validate")
    parser.add_argument("--config", help="Path to configuration JSON file")
    parser.add_argument("--output", help="Output directory for reports")
    args = parser.parse_args()
    
    # Initialize validator
    validator = SceneValidator(args.config)
    
    # Override output directory if specified
    if args.output:
        validator.config["output_dir"] = args.output
    
    # Run validation
    result = validator.validate_video(args.video_path)
    
    # Print summary
    print(f"\nValidation complete!")
    print(f"Found {result.get('total_issues', 0)} issues:")
    
    if 'issue_summary' in result:
        summary = result['issue_summary']
        if 'by_type' in summary:
            print("\nIssues by type:")
            for issue_type, count in summary['by_type'].items():
                print(f"  - {issue_type.capitalize()}: {count}")
        
        if 'by_severity' in summary:
            print("\nIssues by severity:")
            for severity, count in summary['by_severity'].items():
                print(f"  - {severity.capitalize()}: {count}")
    
    print(f"\nDetailed report saved to: {validator.config['output_dir']}")


if __name__ == "__main__":
    main()