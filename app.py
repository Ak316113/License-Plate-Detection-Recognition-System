import os
import logging
import uuid
import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import json

# Import detection and recognition modules
# These will handle any module errors internally
from detector import detect_license_plate
from recognizer import recognize_text

# Import blur/deblur utilities for the visual trick
try:
    # Use the simplified deblur script that strictly matches by image number
    from simple_deblur import deblur_image
    DEBLUR_AVAILABLE = True
    logging.info("Using simplified number-based matching for deblurring")
except ImportError:
    # Fallback options if the simple script isn't available
    try:
        from direct_deblur import deblur_image
        DEBLUR_AVAILABLE = True
        logging.info("Using direct visual blur/deblur system for your specific setup")
    except ImportError:
        try:
            from custom_deblur import deblur_image
            DEBLUR_AVAILABLE = True
            logging.info("Using custom visual blur/deblur trick system")
        except ImportError:
            try:
                from deblur_image import deblur_image
                DEBLUR_AVAILABLE = True
                logging.info("Using original visual blur/deblur trick system")
            except ImportError:
                DEBLUR_AVAILABLE = False
                logging.warning("No deblurring module available, deblurring will be disabled")

# Define a now() function for templates
def now():
    return datetime.datetime.now()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "license-plate-detection-secret")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure SQLite database
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///license_plates.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Configure file uploads
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["CROPPED_FOLDER"] = "static/crops"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "mp4", "avi", "mov"}
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max upload

# Ensure directories exist
for folder in [app.config["UPLOAD_FOLDER"], app.config["CROPPED_FOLDER"]]:
    os.makedirs(folder, exist_ok=True)

# Define the base class for SQLAlchemy models
class Base(DeclarativeBase):
    pass

# Initialize SQLAlchemy
db = SQLAlchemy(model_class=Base)
db.init_app(app)

# Import models after initializing db
from models import Detection

# Create database tables
with app.app_context():
    db.create_all()

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def save_uploaded_file(file):
    """Save the uploaded file and return the saved path"""
    filename = secure_filename(file.filename)
    # Add a UUID to prevent filename collisions
    unique_filename = f"{uuid.uuid4()}_{filename}"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    file.save(file_path)
    logger.debug(f"Saved file to {file_path}")
    return file_path

def is_video(filename):
    """Check if the file is a video based on its extension"""
    video_extensions = {"mp4", "avi", "mov"}
    return filename.rsplit(".", 1)[1].lower() in video_extensions

def extract_frames(video_path, max_frames=10):
    """Extract frames from a video file"""
    try:
        # Import cv2 here to handle import errors gracefully
        import cv2
        
        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Determine frame interval to extract evenly distributed frames
        if total_frames <= max_frames:
            frame_interval = 1
        else:
            frame_interval = total_frames // max_frames
        
        count = 0
        frame_positions = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if count % frame_interval == 0 and len(frames) < max_frames:
                frames.append(frame)
                frame_positions.append(count)
            
            count += 1
        
        cap.release()
        logger.debug(f"Extracted {len(frames)} frames from video")
        return frames, frame_positions
    except ImportError:
        logger.error("OpenCV (cv2) module not available for video processing")
        # Return empty for graceful failure 
        return [], []
    except Exception as e:
        logger.error(f"Error extracting frames: {str(e)}")
        return [], []

@app.route("/")
def index():
    """Render the main page"""
    # Pass the now function to the template
    return render_template("index.html", now=now)

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file upload and processing"""
    if "file" not in request.files:
        flash("No file part", "danger")
        return redirect(request.url)
    
    file = request.files["file"]
    
    if file.filename == "":
        flash("No file selected", "danger")
        return redirect(request.url)
    
    if not allowed_file(file.filename):
        flash("File type not allowed", "danger")
        return redirect(request.url)
    
    try:
        # Check if deblurring is requested
        use_deblur = request.form.get('deblur') == '1'
        if use_deblur and not DEBLUR_AVAILABLE:
            logger.warning("Deblurring requested but not available")
            flash("Deblurring is not available in this environment", "warning")
            use_deblur = False
            
        file_path = save_uploaded_file(file)
        
        detections = []
        
        if is_video(file_path):
            try:
                # Process video file
                import cv2
                frames, frame_positions = extract_frames(file_path)
                for i, frame in enumerate(frames):
                    # Save frame as image
                    frame_filename = f"{uuid.uuid4()}_frame_{i}.jpg"
                    frame_path = os.path.join(app.config["UPLOAD_FOLDER"], frame_filename)
                    cv2.imwrite(frame_path, frame)
                    
                    # Detect plate in the frame
                    detection_results = process_image(frame_path, frame_positions[i], use_deblur)
                    if detection_results:
                        detections.append(detection_results)
            except ImportError:
                logger.error("OpenCV (cv2) module not available for video processing")
                return jsonify({"success": False, "error": "Video processing is not available in this environment"})
        else:
            # Process image file
            detection_results = process_image(file_path, None, use_deblur)
            if detection_results:
                detections.append(detection_results)
        
        # Return detection results
        return jsonify({"success": True, "detections": detections})
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        flash(f"Error processing file: {str(e)}", "danger")
        return jsonify({"success": False, "error": str(e)})

def process_image(image_path, frame_position=None, use_deblur=False):
    """Process an image to detect and recognize license plate"""
    try:
        # Create a directory for deblurred images if needed
        deblurred_dir = os.path.join("static", "deblurred")
        os.makedirs(deblurred_dir, exist_ok=True)
        
        processing_path = image_path
        is_deblurred = False
        
        # Apply deblurring if requested and available
        if use_deblur and DEBLUR_AVAILABLE:
            logger.info(f"Applying deblurring to {image_path}")
            try:
                # Check if this file might already be a blurred image
                base_name = os.path.basename(image_path)
                is_blurred_file = '_blurred' in base_name
                
                # Generate path for deblurred image
                deblurred_filename = f"deblurred_{os.path.basename(image_path)}"
                deblurred_path = os.path.join(deblurred_dir, deblurred_filename)
                
                # Look for parameter file in various locations
                param_file = None
                if is_blurred_file:
                    # Try different parameter file locations
                    possible_param_files = [
                        f"{image_path}.params.json",  # Direct params file
                        os.path.join(os.path.dirname(image_path), 
                                    os.path.splitext(base_name)[0] + ".params.json"),  # Same dir with same name
                        os.path.join(os.path.dirname(image_path), 
                                    os.path.splitext(base_name)[0].replace('_blurred', '') + ".params.json")  # Original name
                    ]
                    
                    for file_path in possible_param_files:
                        if os.path.exists(file_path):
                            param_file = file_path
                            logger.info(f"Found parameter file at: {param_file}")
                            break
                
                # Apply deblurring, but only pass the parameters the function accepts
                deblurred_path = deblur_image(image_path, deblurred_path)
                
                if deblurred_path and os.path.exists(deblurred_path):
                    logger.info(f"Deblurring successful, using deblurred image: {deblurred_path}")
                    processing_path = deblurred_path
                    is_deblurred = True
                else:
                    logger.warning("Deblurring failed, using original image")
            except Exception as e:
                logger.error(f"Error in deblurring: {str(e)}")
                # Continue with original image if deblurring fails
        
        # Detect license plate on the selected image (original or deblurred)
        plate_results = detect_license_plate(processing_path)
        
        if not plate_results:
            logger.warning(f"No license plate detected in {processing_path}")
            return None
        
        detections = []
        for i, (crop_path, confidence, coords) in enumerate(plate_results):
            # Recognize text on the cropped license plate
            ocr_text = recognize_text(crop_path)
            
            # Get relative path for frontend display
            # Always show the original image in the UI, but note if deblurring was applied
            relative_upload_path = os.path.relpath(image_path, start="static")
            relative_crop_path = os.path.relpath(crop_path, start="static")
            
            # Create a detection record
            timestamp = datetime.datetime.now()
            
            # Prepare description
            processing_desc = "Deblurring applied" if is_deblurred else ""
            
            # Create database model instance
            detection = Detection(
                image_path=relative_upload_path,
                plate_crop_path=relative_crop_path,
                plate_text=ocr_text,
                confidence=confidence,
                coords=json.dumps(coords),
                timestamp=timestamp,
                frame_position=frame_position
            )
            
            # Add and commit to database
            db.session.add(detection)
            db.session.commit()
            
            # Prepare detection data
            detection_data = {
                "id": detection.id,
                "image_path": relative_upload_path,
                "plate_crop_path": relative_crop_path,
                "plate_text": ocr_text,
                "confidence": confidence,
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "frame_position": frame_position,
                "deblurred": is_deblurred
            }
            detections.append(detection_data)
        
        return detections
    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}")
        raise

@app.route("/history")
def history():
    """Show detection history"""
    # Get all detections, ordered by timestamp descending
    detections = Detection.query.order_by(Detection.timestamp.desc()).all()
    return render_template("history.html", detections=detections, now=now)

@app.route("/delete/<int:detection_id>", methods=["POST"])
def delete_detection(detection_id):
    """Delete a detection record"""
    detection = Detection.query.get_or_404(detection_id)
    
    try:
        db.session.delete(detection)
        db.session.commit()
        flash("Detection record deleted successfully", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting record: {str(e)}", "danger")
    
    return redirect(url_for("history"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
