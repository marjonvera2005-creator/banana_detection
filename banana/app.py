from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}


HISTORY_FILE = 'database/history.json'

try:
    from ultralytics import YOLO
    import urllib.request
    
    # Try multiple possible paths for the model
    model_paths = ['model/best.pt', 'best.pt', '/opt/render/project/src/model/best.pt']
    model = None
    
    # If no model found, try to download from a URL (replace with your model URL)
    for path in model_paths:
        try:
            if os.path.exists(path):
                model = YOLO(path)
                print(f"Model loaded successfully from: {path}")
                break
        except:
            continue
    
    if model is None:
        print("WARNING: Model not found in any expected location")
        # Uncomment and add your model download URL here:
        # urllib.request.urlretrieve('YOUR_MODEL_URL', 'best.pt')
        # model = YOLO('best.pt')
except Exception as e:
    print(f"ERROR loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_history(data):
    history = load_history()
    history.append(data)
    os.makedirs('database', exist_ok=True)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def get_recommendation(class_name, confidence):
    recommendations = {
        'unripe': {
            'status': 'PASS',
            'color': 'green',
            'message': 'Store for ripening. Not ready for sale.',
            'action': 'Keep in storage'
        },
        'ripe': {
            'status': 'PASS',
            'color': 'yellow',
            'message': 'Perfect condition. Ready for sale.',
            'action': 'Approve for distribution'
        },
        'overripe': {
            'status': 'WARNING',
            'color': 'black',
            'message': 'Sell immediately or use for processing.',
            'action': 'Priority sale'
        },
        'rotten': {
            'status': 'FAIL',
            'color': 'brown',
            'message': 'Quality compromised. Do not sell.',
            'action': 'Discard immediately'
        }
    }
    return recommendations.get(class_name.lower(), recommendations['ripe'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect')
def detect_page():
    return render_template('detect.html')

@app.route('/history')
def history_page():
    history = load_history()
    return render_template('history.html', history=history)

@app.route('/dashboard')
def dashboard_page():
    history = load_history()
    total = len(history)
    stats = {'unripe': 0, 'ripe': 0, 'overripe': 0, 'rotten': 0}
    
    for item in history:
        class_name = item['class'].lower()
        if class_name in stats:
            stats[class_name] += 1
    
    return render_template('dashboard.html', stats=stats, total=total)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 400
        
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"File saved to: {filepath}")
        print("Starting prediction...")
        results = model(filepath)
        result = results[0]
        
        detections = []
        stats = {'unripe': 0, 'ripe': 0, 'overripe': 0, 'rotten': 0}
        
        # Check if it's a detection model (has boxes)
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            if len(boxes) > 0:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = result.names[class_id]
                    coords = box.xyxy[0].tolist()
                    
                    detections.append({
                        'class': class_name,
                        'confidence': round(confidence * 100, 2),
                        'coords': coords
                    })
                    
                    class_lower = class_name.lower()
                    if class_lower in stats:
                        stats[class_lower] += 1
        # Check if it's a classification model (has probs)
        elif hasattr(result, 'probs') and result.probs is not None:
            class_id = result.probs.top1
            confidence = float(result.probs.top1conf)
            class_name = result.names[class_id]
            
            detections.append({
                'class': class_name,
                'confidence': round(confidence * 100, 2),
                'coords': [0, 0, 100, 100]
            })
            
            class_lower = class_name.lower()
            if class_lower in stats:
                stats[class_lower] += 1
        
        if not detections:
            return jsonify({'success': False, 'error': 'No banana detected'}), 400
        
        top_detection = detections[0]
        recommendation = get_recommendation(top_detection['class'], top_detection['confidence'] / 100)
        
        history_data = {
            'image': filename,
            'class': top_detection['class'],
            'confidence': top_detection['confidence'],
            'status': recommendation['status'],
            'recommendation': recommendation['message'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        save_history(history_data)
        
        response = {
            'success': True,
            'class': top_detection['class'],
            'confidence': top_detection['confidence'],
            'status': recommendation['status'],
            'color': recommendation['color'],
            'message': recommendation['message'],
            'action': recommendation['action'],
            'image': filename,
            'detections': detections,
            'stats': stats,
            'total_detected': len(detections)
        }
        print(f"Success: {response}")
        return jsonify(response), 200
    except Exception as e:
        import traceback
        print(f"Route error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    print(f"Exception: {e}")
    import traceback
    traceback.print_exc()
    return jsonify({'success': False, 'error': f'Unexpected error: {str(e)}'}), 500

@app.errorhandler(500)
def handle_500(e):
    print(f"500 Error: {e}")
    import traceback
    traceback.print_exc()
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

@app.errorhandler(404)
def handle_404(e):
    return jsonify({'success': False, 'error': 'Not found'}), 404

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('database', exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    print("=" * 50)
    print("Starting Banana Inspection System...")
    print(f"Model loaded: {model is not None}")
    print(f"Port: {port}")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=port)
