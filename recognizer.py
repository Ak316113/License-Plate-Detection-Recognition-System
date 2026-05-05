import os
import logging
import cv2
import random
import string

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# EasyOCR reader - will be initialized if available
reader = None

def preprocess_plate(image):
    """
    Improve OCR accuracy with basic preprocessing
    This function matches the preprocessing in Recognition easyocr.py
    """
    try:
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not load image: {image}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to preserve edges
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Adaptive thresholding for contrast
        thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 15, 10)
        
        # Convert back to BGR for EasyOCR (which expects color images)
        processed = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        return processed
    
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        return image

def recognize_text(image_path):
    """
    Recognize text in license plate image using EasyOCR
    
    This function implements license plate OCR using EasyOCR.
    For local implementation, this should be run with EasyOCR installed.
    
    Args:
        image_path: Path to the cropped license plate image
    
    Returns:
        Recognized text as a string
    """
    try:
        # Try to use EasyOCR if available (for local implementation)
        import easyocr
        global reader
        
        # Initialize the EasyOCR reader if needed (only once)
        if reader is None:
            reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if available
            logger.debug("EasyOCR reader initialized")
        
        # Load and preprocess the image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return "ERROR: Failed to load image"
        
        # Preprocess the image for better OCR results
        processed = preprocess_plate(image)
        
        # Run EasyOCR on the processed image
        results = reader.readtext(processed)
        
        # Extract and clean the recognized text
        text = ''
        for bbox, detected_text, confidence in results:
            # Keep only alphanumeric characters
            cleaned = ''.join(filter(str.isalnum, detected_text))
            text += cleaned + ' '
        
        # Final result
        final_text = text.strip()
        logger.info(f"Recognized license plate text: {final_text}")
        return final_text
        
    except ImportError:
        # Fall back to mock implementation if EasyOCR is not available
        logger.warning("EasyOCR not available - using mock OCR for development only")
        return mock_recognize_text(image_path)
    except Exception as e:
        logger.error(f"Error in OCR: {str(e)}")
        return f"ERROR: {str(e)}"

def mock_recognize_text(image_path):
    """
    Generate mock license plate text for development purposes.
    This is ONLY used when EasyOCR is not available.
    
    For actual deployment, the real implementation above should be used.
    """
    try:
        # Generate realistic mock license plate formats
        formats = [
            # Format: 2 letters, 3 numbers, 2 letters (UK style)
            lambda: ''.join(random.choices(string.ascii_uppercase, k=2)) + 
                    ''.join(random.choices(string.digits, k=3)) + 
                    ''.join(random.choices(string.ascii_uppercase, k=2)),
            # Format: 3 letters, 3 numbers (US style)
            lambda: ''.join(random.choices(string.ascii_uppercase, k=3)) + 
                    ''.join(random.choices(string.digits, k=3)),
            # Format: 2 letters, 2 digits, 2 letters (European style)
            lambda: ''.join(random.choices(string.ascii_uppercase, k=2)) + 
                    ''.join(random.choices(string.digits, k=2)) + 
                    ''.join(random.choices(string.ascii_uppercase, k=2)),
        ]
        
        # Choose a random format and generate plate text
        mock_plate_text = random.choice(formats)()
        
        logger.info(f"Generated mock license plate text: {mock_plate_text}")
        return mock_plate_text
        
    except Exception as e:
        logger.error(f"Error generating mock text: {str(e)}")
        return "ERROR"
