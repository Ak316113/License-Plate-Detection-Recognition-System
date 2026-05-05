# License Plate Detection and Recognition System

A web application that detects and recognizes license plates from uploaded images and videos using YOLOv8 and EasyOCR.

## Features

- Upload and process images or videos containing license plates
- Automatic detection of license plates using YOLOv8
- Text recognition from license plates using EasyOCR
- History view of all detected license plates
- Reversible blurring capability for preprocessing

## Project Structure

- `/static` - Static files (images, CSS, JS)
- `/templates` - HTML templates
- `/doc` - Project documentation
- `app.py` - Main Flask application
- `detector.py` - License plate detection using YOLOv8
- `recognizer.py` - License plate text recognition using EasyOCR
- `models.py` - Database models
- `enhanced_blur.py` - Reversible blurring utility

## Usage

1. Start the application:
   ```
   python main.py
   ```

2. Open a browser and navigate to `http://localhost:5000`

3. Upload an image or video containing license plates

4. View the detection results

5. Use the "Apply deblurring" option to process blurred images

## Documentation

Detailed documentation is available in the `/doc` directory:

- `model_details.md` - Technical details about the YOLOv8 model
- `project_synopsis.md` - Overview of the project and its capabilities
- `project_report.md` - Comprehensive project report
- `usage_guide.md` - User guide for the application

## Blurring & Deblurring

The system includes a special utility for creating and reversing blurred images:

```python
from enhanced_blur import reversible_blur, recover_from_blur

# To blur an image
reversible_blur("path/to/original.jpg", "path/to/blurred.jpg", blur_strength=3)

# To recover the original
recover_from_blur("path/to/blurred.jpg", "path/to/recovered.jpg")
```

When uploading blurred images through the web interface, enable the "Apply deblurring before detection and recognition" option to process them.