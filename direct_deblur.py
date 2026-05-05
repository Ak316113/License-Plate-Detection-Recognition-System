import cv2
import os
import sys
import logging
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deblur_image(blurred_path, output_path=None):
    """
    A direct deblurring function specifically designed to find original images
    for files uploaded by the user.
    
    This function:
    1. Looks for corresponding original images in the user's directories
    2. Creates a copy in the deblurred folder if found
    3. Falls back to the blurred image if original can't be found
    
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
            
        # Get the base filename without _blurred suffix (if present)
        base_name, extension = os.path.splitext(filename)
        clean_name = base_name
        if "_blurred" in base_name:
            clean_name = base_name.replace("_blurred", "")
            
        # The user specific directory 
        user_blurr_dir = "E:\\4th Sem\\Road Safty Management\\Trained Model YOLO\\Blurr Deblurr\\LicensePlateTracker\\blurr_images"
        
        # Extract important parts of the filename to find the matching original
        # First, identify what kind of image this is
        
        # The core of the deblurring issue is here:
        # We need to match each blurred image with its SPECIFIC original image
        
        # Identify the base file pattern - for example, if this is "24_blurred.jpg"
        # we need to look for "24_original.jpg", not a different image
        
        image_identifier = None
        
        # Check if this is one of your numbered images (like "24_blurred.jpg")
        if "_blurred" in base_name:
            clean_base = base_name.replace("_blurred", "")
            # This might be something like "24" or "UUID_24" depending on your naming
            for part in clean_base.split("_"):
                if part.isdigit() or (len(part) <= 3 and part.isalnum()):
                    # This is likely the image number identifier (like "24") 
                    image_identifier = part
                    logger.info(f"Found image identifier: {image_identifier}")
                    break
        
        # If we found a clear image number, prioritize looking for that specific image
        specific_original_paths = []
        
        if image_identifier:
            # Look for originals that match this specific image identifier
            specific_original_paths = [
                # Direct matches with the image identifier
                os.path.join(user_blurr_dir, f"{image_identifier}_original{extension}"),
                os.path.join(user_blurr_dir, f"{image_identifier}_original.jpg"),
                os.path.join(user_blurr_dir, f"{image_identifier}_original.jpeg"),
                os.path.join(user_blurr_dir, f"{image_identifier}.jpg"),
                os.path.join(user_blurr_dir, f"{image_identifier}.jpeg"),
                
                # The specific original name without UUID
                os.path.join(user_blurr_dir, f"original_{image_identifier}{extension}"),
                os.path.join(user_blurr_dir, f"original_{image_identifier}.jpg"),
                
                # Try looking in the parent directory too
                os.path.join(os.path.dirname(user_blurr_dir), "blurr_images", f"{image_identifier}_original{extension}"),
                os.path.join(os.path.dirname(user_blurr_dir), "blurr_images", f"{image_identifier}.jpg"),
            ]
        
        # Always check for exact matching original for this specific blurred file
        base_possible_paths = [
            # Base paths with full filename
            os.path.join(user_blurr_dir, f"{clean_name}_original{extension}"),
            os.path.join(user_blurr_dir, f"{clean_name}{extension}"),
            os.path.join(user_blurr_dir, f"original_{clean_name}{extension}"),
        ]
        
        # Get the original image name without UUIDs (for additional matches)
        # Look for patterns like UUID___actualFileName.jpg_blurred.jpeg
        actual_file_name = None
        if "___" in base_name:
            # Format is usually: UUID___actualFileName
            parts = base_name.split("___", 1)
            if len(parts) > 1:
                actual_file_name = parts[1]
                # If it ends with _blurred, remove that part
                if "_blurred" in actual_file_name:
                    actual_file_name = actual_file_name.replace("_blurred", "")
                logger.info(f"Extracted actual file name: {actual_file_name}")
        
        # If we have an actual file name, add those paths too
        actual_file_paths = []
        if actual_file_name:
            actual_file_paths = [
                # Paths with the actual filename
                os.path.join(user_blurr_dir, f"{actual_file_name}_original{extension}"),
                os.path.join(user_blurr_dir, f"{actual_file_name}_original.jpg"),
                os.path.join(user_blurr_dir, f"{actual_file_name}_original.jpeg"),
                os.path.join(user_blurr_dir, f"{actual_file_name}"),
                os.path.join(user_blurr_dir, f"{actual_file_name}.jpg"),
                os.path.join(user_blurr_dir, f"{actual_file_name}.jpeg"),
                
                # Try versions without file extension in the name
                os.path.join(user_blurr_dir, actual_file_name.split('.')[0] + "_original" + extension),
                os.path.join(user_blurr_dir, actual_file_name.split('.')[0] + "_original.jpg"),
                
                # Try with different folder structures
                os.path.join("blurr_images", f"{actual_file_name}_original{extension}"),
                os.path.join("blurr_images", f"{actual_file_name}"),
            ]
        
        # Combine all paths, prioritizing the most specific matches first
        possible_paths = []
        
        # 1. First try paths based on image identifier (highest priority)
        if image_identifier:
            possible_paths.extend(specific_original_paths)
            
        # 2. Then try exact matching paths for this file
        possible_paths.extend(base_possible_paths)
        
        # 3. Then try paths based on actual file name
        if actual_file_name:
            possible_paths.extend(actual_file_paths)
        
        # Log which image identifier we're looking for to help with debugging
        if image_identifier:
            logger.info(f"Looking for original image matching identifier: {image_identifier}")
        elif actual_file_name:
            logger.info(f"Looking for original image matching name: {actual_file_name}")
        
        # Avoid using hardcoded fallbacks that can cause incorrect matches
        # The specific image "15e7c19f-da3c-4f0d-91f3-552afabc8961___41FxrEl7jHL.jpg" 
        # should only be used when that exact image is being processed
        
        # Log the paths we're checking for debugging
        logger.info(f"Looking for original image with these possible paths:")
        for path in possible_paths[:5]:  # Just log a few to avoid flooding
            logger.info(f"- {path}")
            
        # Try each path
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found original image at: {path}")
                # If path exists, copy to output
                shutil.copy2(path, output_path)
                logger.info(f"Copied original to: {output_path}")
                return output_path
                
        # If we get here, no original image was found
        # As a fallback, we'll just use the blurred image
        logger.warning(f"No original image found for: {filename}, using the blurred image itself")
        shutil.copy2(blurred_path, output_path)
        return output_path
        
    except Exception as e:
        logger.error(f"Error in deblur_image: {str(e)}")
        # As a last resort fallback, return the blurred path
        return blurred_path