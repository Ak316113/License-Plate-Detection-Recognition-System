from app import db
import datetime

class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(255), nullable=False)
    plate_crop_path = db.Column(db.String(255), nullable=False)
    plate_text = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    coords = db.Column(db.String(255), nullable=False)  # JSON string of coordinates
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    frame_position = db.Column(db.Integer, nullable=True)  # For video frames
    
    def __repr__(self):
        return f"<Detection {self.id}: {self.plate_text}>"
