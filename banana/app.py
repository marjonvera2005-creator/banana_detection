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
    model = YOLO('model/best.pt')
    print("Model loaded successfully!")
except:
    print("WARNING: Model not found. Add best.pt to model/ folder")
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
            'color': 'yellow',
            'message': 'Store for ripening. Not ready for sale.',
            'action': 'Keep in storage'
        },
        'ripe': {
            'status': 'PASS',
            'color': 'green',
            'message': 'Perfect condition. Ready for sale.',
            'action': 'Approve for distribution'
        },
        'overripe': {
            'status': 'WARNING',
            'color': 'orange',
            'message': 'Sell immediately or use for processing.',
            'action': 'Priority sale'
        },
        'rotten': {
            'status': 'FAIL',
            'color': 'red',
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
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please add best.pt to model/ folder'
        })
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            results = model(filepath)
            result = results[0]
            
            if hasattr(result, 'probs'):
                class_id = result.probs.top1
                confidence = float(result.probs.top1conf)
                class_name = result.names[class_id]
            else:
                if len(result.boxes) > 0:
                    class_id = int(result.boxes[0].cls[0])
                    confidence = float(result.boxes[0].conf[0])
                    class_name = result.names[class_id]
                else:
                    return jsonify({'success': False, 'error': 'No banana detected'})
            
            recommendation = get_recommendation(class_name, confidence)
            
            history_data = {
                'image': filename,
                'class': class_name,
                'confidence': round(confidence * 100, 2),
                'status': recommendation['status'],
                'recommendation': recommendation['message'],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            save_history(history_data)
            
            return jsonify({
                'success': True,
                'class': class_name,
                'confidence': round(confidence * 100, 2),
                'status': recommendation['status'],
                'color': recommendation['color'],
                'message': recommendation['message'],
                'action': recommendation['action'],
                'image': filename
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    return jsonify({'success': False, 'error': 'Invalid file type'})

@app.errorhandler(500)
def handle_500(e):
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
    print(f"Port: {port}")
    print("=" * 50)
    app.run(debug=False, host='0.0.0.0', port=port)
