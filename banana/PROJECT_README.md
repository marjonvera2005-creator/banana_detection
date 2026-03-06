# YOLO-Based Visual Inspection for Banana Ripeness and Defect Detection

## 🎯 Project Overview

Automated banana quality inspection system using YOLOv8 deep learning technology to classify banana ripeness (Unripe, Ripe, Overripe, Rotten) with high accuracy.

## 🚀 Features

- ✅ Image Upload Inspection
- ✅ Real-Time Camera Detection
- ✅ Automatic Quality Classification
- ✅ Confidence Score Display
- ✅ Inspection History Logging
- ✅ Statistical Dashboard
- ✅ Quality Recommendation System
- ✅ Admin Monitoring Panel

## 📁 Project Structure

```
banana-inspection/
├── app.py                    # Main Flask application
├── model/
│   └── best.pt              # Trained YOLOv8 model (add after training)
├── static/
│   ├── css/
│   ├── js/
│   └── uploads/             # Uploaded images
├── templates/
│   ├── index.html           # Home page
│   ├── detect.html          # Detection page
│   ├── history.html         # History page
│   └── dashboard.html       # Dashboard page
├── database/
│   └── history.json         # Inspection records
└── requirements.txt         # Python dependencies
```

## 🛠️ Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Add Your Trained Model

1. Train your model in Google Colab
2. Download `best.pt`
3. Place it in `model/best.pt`

### Step 3: Run the Application

```bash
python app.py
```

### Step 4: Open in Browser

```
http://127.0.0.1:5000
```

## 📊 Classes

- **Unripe** - Early stage, store for ripening
- **Ripe** - Perfect condition, ready for sale
- **Overripe** - Urgent sale needed
- **Rotten** - Quality compromised, discard

## 🎓 Technology Stack

- **Backend:** Flask (Python)
- **AI Model:** YOLOv8 (Ultralytics)
- **Frontend:** HTML, Tailwind CSS, JavaScript
- **Database:** JSON file storage
- **Charts:** Chart.js

## 📝 Usage

1. **Home Page** - View system overview
2. **Detection Page** - Upload image for inspection
3. **History Page** - View all past inspections
4. **Dashboard Page** - View statistics and charts

## 🔬 Model Training

See `train.py` and `train_detection.py` for training scripts.

## 📄 License

Educational Project - Thesis/Capstone

## 👥 Authors

Your Name / Team Name
