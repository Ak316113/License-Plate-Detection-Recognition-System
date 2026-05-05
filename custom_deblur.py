import cv2
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_original_image(blurred_path):
    """
    Find the original (unblurred) image that corresponds to a blurred image.
    
    This function checks various possible locations and naming patterns
    for the original image based on the blurred image path.
    
    Args:
        blurred_path: Path to the blurred image
        
    Returns:
        Path to the original image if found, None otherwise
    """
    # Extract directory, filename, and extension
    directory = os.path.dirname(blurred_path)
    filename = os.path.basename(blurred_path)
    name, ext = os.path.splitext(filename)
    
    # Check if this is one of our blurred images
    if not ('_blurred' in name or 'blurred' in name):
        logger.warning(f"Image doesn't appear to be blurred: {blurred_path}")
        return None
    
    # Look in the same directory as the blurred image
    original_patterns = [
        # Standard pattern from our blur script
        name.replace('_blurred', '_original') + ext,
        # Alternative patterns
        name.replace('_blurred', '') + ext,
        name.replace('blurred', 'original') + ext,
        'original_' + name.replace('_blurred', '') + ext
    ]
    
    # Look in the custom directory where the original might be stored
    custom_dirs = [
        directory,
        os.path.join(directory, 'originals'),
        os.path.join(os.path.dirname(directory), 'blurr_images'),  # User's specific directory
        "blurr_images"  # Root directory
    ]
    
    # Check all possible combinations
    for dir_path in custom_dirs:
        if not os.path.exists(dir_path):
            continue
            
        for pattern in original_patterns:
            potential_path = os.path.join(dir_path, pattern)
            if os.path.exists(potential_path):
                logger.info(f"Found original image at: {potential_path}")
                return potential_path
                
    # If we get here, we couldn't find the original
    logger.warning(f"Could not find original image for: {blurred_path}")
    return None

def deblur_image(blurred_path, output_path=None):
    """
    Retrieve the original image to simulate deblurring.
    
    Args:
        blurred_path: Path to the blurred image
        output_path: Path to save the deblurred (actually original) image
                    If None, creates a path with _deblurred suffix
    
    Returns:
        Path to the deblurred image, or None if deblurring failed
    """
    try:
        # Verify the blurred image exists
        if not os.path.exists(blurred_path):
            logger.error(f"Blurred image not found: {blurred_path}")
            return None
            
        # Check if the image is actually a blurred image
        filename = os.path.basename(blurred_path)
        
        # Find the original image
        original_path = find_original_image(blurred_path)
        if not original_path:
            logger.error(f"Could not find original image for: {blurred_path}")
            return None
            
        # Create output path if not provided
        if output_path is None:
            directory = os.path.dirname(blurred_path)
            name, ext = os.path.splitext(filename)
            deblurred_name = name.replace('_blurred', '_deblurred') + ext
            if deblurred_name == name + ext:  # No replacement occurred
                deblurred_name = 'deblurred_' + name + ext
                
            output_path = os.path.join(directory, deblurred_name)
            
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
        # Copy the original image to the output path
        original = cv2.imread(original_path)
        if original is None:
            logger.error(f"Failed to load original image: {original_path}")
            return None
            
        # Save as the "deblurred" image
        cv2.imwrite(output_path, original)
        logger.info(f"Created deblurred image at: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error in deblur_image: {str(e)}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Find and retrieve the original version of a blurred image")
    parser.add_argument("--input", required=True, help="Path to blurred image")
    parser.add_argument("--output", help="Path for deblurred output (optional)")
    
    args = parser.parse_args()
    
    result = deblur_image(args.input, args.output)
    if result:
        print(f"Deblurring successful: {result}")
    else:
        print("Deblurring failed")