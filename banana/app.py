import gradio as gr
from ultralytics import YOLO
from PIL import Image

# Load model
try:
    model = YOLO('model/best.pt')
    print("Model loaded successfully!")
except:
    print("Model not found")
    model = None

def get_recommendation(class_name, confidence):
    recommendations = {
        'unripe': {
            'status': 'PASS',
            'color': '🟢',
            'message': 'Store for ripening. Not ready for sale.',
            'action': 'Keep in storage'
        },
        'ripe': {
            'status': 'PASS', 
            'color': '🟡',
            'message': 'Perfect condition. Ready for sale.',
            'action': 'Approve for distribution'
        },
        'overripe': {
            'status': 'WARNING',
            'color': '⚫',
            'message': 'Sell immediately or use for processing.',
            'action': 'Priority sale'
        },
        'rotten': {
            'status': 'FAIL',
            'color': '🟤',
            'message': 'Quality compromised. Do not sell.',
            'action': 'Discard immediately'
        }
    }
    return recommendations.get(class_name.lower(), recommendations['ripe'])

def predict_banana(image):
    if model is None:
        return "❌ Model not loaded"
    
    try:
        results = model(image)
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
                return "❌ No banana detected"
        
        recommendation = get_recommendation(class_name, confidence)
        
        output = f"""
{recommendation['color']} **{class_name.upper()}**

**Confidence:** {confidence*100:.1f}%
**Status:** {recommendation['status']}
**Recommendation:** {recommendation['message']}
**Action:** {recommendation['action']}
        """
        
        return output
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

iface = gr.Interface(
    fn=predict_banana,
    inputs=gr.Image(type="pil"),
    outputs=gr.Markdown(),
    title="🍌 Banana Ripeness Detection System",
    description="Upload a banana image to detect its ripeness level using YOLOv8"
)

if __name__ == "__main__":
    iface.launch()
