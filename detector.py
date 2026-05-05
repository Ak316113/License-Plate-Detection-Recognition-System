import os
import uuid
import logging
import subprocess
import json
import cv2

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Paths and directories
MODEL_PATH = "attached_assets/best.pt"  # Path to YOLOv8 weights file
CROP_FOLDER = "static/crops"  # Folder to save cropped license plate images

# Ensure crop folder exists
os.makedirs(CROP_FOLDER, exist_ok=True)

def detect_license_plate(image_path):
    """
    Detect license plates in the provided image using YOLOv8
    
    This function uses the YOLOv8 model to detect license plates in the image.
    For local implementation, this should be configured to use the actual model.
    
    Returns:
        List of tuples: (cropped_image_path, confidence, coordinates)
    """
    try:
        # Import YOLOv8 - this is for local implementation
        from ultralytics import YOLO
        
        # Load the YOLOv8 model with the provided weights
        model = YOLO(MODEL_PATH)
        logger.debug(f"YOLO model loaded successfully from {MODEL_PATH}")
        
        # Perform detection
        results = model(image_path)
        
        plate_results = []
        
        # Process the detection results
        for i, result in enumerate(results):
            # Loop through each detected box
            for j, box in enumerate(result.boxes):
                # Extract coordinates, confidence, and class
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = result.names[cls]
                
                logger.debug(f"Detected {class_name} with confidence {conf:.4f} at ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
                
                # Load the image for cropping
                image = cv2.imread(image_path)
                if image is None:
                    logger.error(f"Failed to load image {image_path}")
                    continue
                
                # Convert coordinates to integers for cropping
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Crop the license plate region
                plate_crop = image[y1:y2, x1:x2]
                
                # Generate a unique filename for the cropped plate
                crop_filename = f"{uuid.uuid4()}_plate_{j}.jpg"
                crop_path = os.path.join(CROP_FOLDER, crop_filename)
                
                # Save the cropped plate
                cv2.imwrite(crop_path, plate_crop)
                logger.debug(f"Plate cropped and saved to {crop_path}")
                
                # Add this detection to results
                plate_results.append((crop_path, conf, (x1, y1, x2, y2)))
        
        return plate_results
        
    except ImportError:
        # For development in Replit where ultralytics might not be available
        logger.warning("YOLO not available - using mock detection for development only")
        return mock_detection_for_development(image_path)
    except Exception as e:
        logger.error(f"Error in license plate detection: {str(e)}")
        return []

def mock_detection_for_development(image_path):
    """
    Mock implementation for development in environments without YOLOv8.
    This is ONLY used when YOLOv8 is not available.
    
    For actual deployment, the real implementation above should be used.
    """
    try:
        import random
        
        # Load the original image to get dimensions
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image {image_path}")
            return []
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Generate 1-2 mock detections
        num_plates = random.randint(1, 2)
        plate_results = []
        
        for i in range(num_plates):
            # Generate a realistic license plate region
            plate_width = width // 6
            plate_height = plate_width // 3
            
            # Position in the lower half of the image
            x1 = random.randint(width // 4, width - width // 4 - plate_width)
            y1 = random.randint(height // 2, height - plate_height - 20)
            x2 = x1 + plate_width
            y2 = y1 + plate_height
            
            # Generate a confidence score
            confidence = random.uniform(0.7, 0.95)
            
            logger.debug(f"Mock detection with confidence {confidence} at ({x1}, {y1}, {x2}, {y2})")
            
            # Crop the region
            plate_crop = image[y1:y2, x1:x2]
            
            # Save the cropped region
            crop_filename = f"{uuid.uuid4()}_plate_{i}.jpg"
            crop_path = os.path.join(CROP_FOLDER, crop_filename)
            cv2.imwrite(crop_path, plate_crop)
            
            # Add to results
            plate_results.append((crop_path, confidence, (x1, y1, x2, y2)))
        
        return plate_results
        
    except Exception as e:
        logger.error(f"Error in mock detection: {str(e)}")
        return []
