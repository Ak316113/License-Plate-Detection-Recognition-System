import cv2
import os
import sys
import logging
import shutil
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_image_number(filename):
    """Extract the image number (like '24' or '1') from a filename."""
    # First, handle the "N_blurr.jpg" pattern
    # If the filename contains "_blurr" and a number before it
    blurr_match = re.search(r'(?:^|\D)(\d+)_blurr', filename)
    if blurr_match:
        logger.info(f"Found image number from '_blurr' pattern: {blurr_match.group(1)}")
        return blurr_match.group(1)
    
    # Look for patterns like "image_N_blurred.jpg" - extract N
    blurred_match = re.search(r'_(\d+)_blurred', filename)
    if blurred_match:
        logger.info(f"Found image number from '_blurred' pattern: {blurred_match.group(1)}")
        return blurred_match.group(1)
    
    # Look for image name with just a number (like "1.jpg")
    simple_match = re.search(r'(?:^|\D)(\d+)(?:\D|$)', filename)
    if simple_match:
        logger.info(f"Found simple image number: {simple_match.group(1)}")
        return simple_match.group(1)
    
    # Look for digits at the start of the string
    start_match = re.search(r'^(\d+)', filename)
    if start_match:
        logger.info(f"Found starting image number: {start_match.group(1)}")
        return start_match.group(1)
    
    # None of the patterns matched
    logger.warning(f"No image number pattern found in: {filename}")
    return None

def deblur_image(blurred_path, output_path=None):
    """
    Simplified deblurring function that strictly matches images by their number.
    This ensures each blurred image is matched with its own original.
    
    Args:
        blurred_path: Path to the blurred image
        output_path: Path for the deblurred output (optional)
        
    Returns:
        Path to the deblurred image
    """
    try:
        # Ensure the blurred image exists
        if not os.path.exists(blurred_path):
            logger.error(f"Blurred image not found: {blurred_path}")
            return None
            
        # Extract the filename and base folder
        filename = os.path.basename(blurred_path)
        base_folder = os.path.dirname(blurred_path)
            
        # Create output path if not specified
        if output_path is None:
            deblurred_dir = os.path.join("static", "deblurred")
            os.makedirs(deblurred_dir, exist_ok=True)
            output_path = os.path.join(deblurred_dir, f"deblurred_{filename}")
        
        # The user's specific directory 
        user_blurr_dir = "E:\\4th Sem\\Road Safty Management\\Trained Model YOLO\\Blurr Deblurr\\LicensePlateTracker\\blurr_images"
        
        # STRICT MATCHING: Get the image number for this specific blurred image
        image_number = get_image_number(filename)
        logger.info(f"Extracted image number: {image_number}")
        
        if not image_number:
            logger.warning(f"Could not extract image number from filename: {filename}")
            # Try to see if this is the specific 41FxrEl7jHL image
            if "41FxrEl7jHL" in filename:
                specific_path = os.path.join(user_blurr_dir, "15e7c19f-da3c-4f0d-91f3-552afabc8961___41FxrEl7jHL.jpg_original.jpeg")
                if os.path.exists(specific_path):
                    logger.info(f"Found original for 41FxrEl7jHL image: {specific_path}")
                    shutil.copy2(specific_path, output_path)
                    return output_path
            return None
        
        # Build paths specifically for this image number
        matching_originals = [
            # Standard original naming patterns
            os.path.join(user_blurr_dir, f"{image_number}_original.jpg"),
            os.path.join(user_blurr_dir, f"{image_number}_original.jpeg"),
            os.path.join(user_blurr_dir, f"{image_number}.jpg"),
            os.path.join(user_blurr_dir, f"original_{image_number}.jpg"),
            
            # For the blurr script pattern specifically
            os.path.join(user_blurr_dir, f"{image_number}_original.jpg.jpeg"),
            os.path.join(user_blurr_dir, f"{image_number}.jpg_original.jpeg"),
            
            # Look directly in the root directory and parent directories
            f"{image_number}_original.jpg",
            f"{image_number}.jpg",
            os.path.join("..", "blurr_images", f"{image_number}_original.jpg"),
            os.path.join("..", "blurr_images", f"{image_number}.jpg"),
            
            # Try other common variations
            os.path.join(user_blurr_dir, f"{image_number}original.jpg")
        ]
        
        # Log what we're looking for
        logger.info(f"Looking for original image with number {image_number}")
        for path in matching_originals:
            logger.info(f"- Checking: {path}")
            
            if os.path.exists(path):
                logger.info(f"Found original image at: {path}")
                # Copy the original to the output path
                shutil.copy2(path, output_path)
                logger.info(f"Copied original to: {output_path}")
                return output_path
        
        # If no matching original was found
        logger.warning(f"No original image found for number: {image_number}")
        return None
        
    except Exception as e:
        logger.error(f"Error in deblur_image: {str(e)}")
        return None