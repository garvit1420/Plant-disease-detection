from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = ('png', 'jpg', 'jpeg')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Disease remedies
disease_remedies = {
    "Apple Scab": {
        "description": "Fungal disease causing dark, scaly lesions on leaves and fruit",
        "remedies": [
            "Apply fungicides containing myclobutanil or sulfur",
            "Remove and destroy infected leaves and fruit",
            "Improve air circulation through pruning",
            "Apply copper-based sprays in early spring"
        ],
        "prevention": "Plant resistant varieties, avoid overhead watering"
    },
    "Apple Black Rot": {
        "description": "Fungal disease causing concentric rings on fruit and leaf spots",
        "remedies": [
            "Apply fungicides with captan or thiophanate-methyl",
            "Remove all mummified fruit from trees",
            "Prune out infected branches 12 inches below cankers",
            "Apply lime sulfur during dormant season"
        ],
        "prevention": "Proper sanitation, remove dead wood regularly"
    },
    "Healthy": {
        "description": "Healthy",
        "remedies": [],
        "prevention": "None"
    }
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_fresh_model():
    """Create a new model with better architecture"""
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', 
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

# Load or create model with enhanced verification
try:
    model = load_model('models/apple_model.h5')
    print("Model loaded successfully")
    
    # Enhanced model verification
    test_cases = [
        ("Random noise", np.random.rand(224,224,3)),
        ("Black image", np.zeros((224,224,3))),
        ("White image", np.ones((224,224,3))),
        ("Red dominant", np.full((224,224,3), [0.9,0.1,0.1])),
        ("Green dominant", np.full((224,224,3), [0.1,0.9,0.1])),
        ("Blue dominant", np.full((224,224,3), [0.1,0.1,0.9]))
    ]
    
    print("\nModel Behavior Verification:")
    biased = False
    for name, img in test_cases:
        processed = np.expand_dims(img.astype('float32'), axis=0)
        pred = model.predict(processed)[0]
        print(f"{name:15} -> Scab: {pred[0]:.2f}, Rot: {pred[1]:.2f}, Rust: {pred[2]:.2f}")
        if pred[2] > 0.8:  # If Rust prediction is too high
            biased = True
    
    if biased:
        print("\nWARNING: Model shows bias toward Healthy - creating fresh model")
        model = create_fresh_model()
        print("New model created. Please retrain with balanced data.")
    else:
        print("\nModel verification passed")
        
except Exception as e:
    print(f"Model loading failed: {e}")
    model = create_fresh_model()
    print("Created fresh model")

# Class names
class_names = {
    0: "Apple Scab",
    1: "Apple Black Rot", 
    2: "Healthy"
}

def preprocess_image(image_path):
    """Enhanced preprocessing with diagnostics"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print("Error: Could not read image file")
            return None
            
        # Convert color space
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize with aspect ratio preservation
        h, w = img.shape[:2]
        if h != w:
            size = max(h, w)
            pad_h = (size - h) // 2
            pad_w = (size - w) // 2
            img = np.pad(img, ((pad_h, size-h-pad_h), (pad_w, size-w-pad_w), (0,0)), 
                        mode='constant')
        img = cv2.resize(img, (224,224))
        
        # Basic quality checks
        if img.mean() < 10 or img.mean() > 245:
            print("Warning: Image may be too dark/bright")
        
        # Normalize
        img = img.astype('float32') / 255.0
        
        # Verify no invalid values
        if np.isnan(img).any() or np.isinf(img).any():
            print("Error: Image contains invalid pixel values")
            return None
            
        return np.expand_dims(img, axis=0)
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None

@app.route('/', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
            
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            processed = preprocess_image(filepath)
            if processed is None:
                return render_template('index.html', 
                                    error="Invalid image - please try another",
                                    result=None)
            
            # Get predictions with timing
            preds = model.predict(processed, verbose=0)[0]
            print("\nPrediction Analysis:")
            print(f"Scab: {preds[0]:.4f}, Rot: {preds[1]:.4f}, Rust: {preds[2]:.4f}")
            
            # Advanced bias correction
            if np.max(preds) > 0.6:  # More sensitive threshold
                print("Applying bias correction")
                # Softmax temperature scaling
                temperature = 2.0  # Higher value flattens predictions more
                scaled = np.log(preds + 1e-10) / temperature
                preds = np.exp(scaled - np.max(scaled))
                preds = preds / preds.sum()
                print(f"After correction - Scab: {preds[0]:.4f}, Rot: {preds[1]:.4f}, Rust: {preds[2]:.4f}")
            
            class_idx = np.argmax(preds)
            confidence = preds[class_idx] * 100
            disease_info = disease_remedies.get(class_names[class_idx], {})
            
            return render_template('index.html',
                                result={
                                    'class': class_names[class_idx],
                                    'confidence': f"{confidence:.1f}%",
                                    'all_predictions': {
                                        'Scab': f"{preds[0]*100:.1f}%",
                                        'Black Rot': f"{preds[1]*100:.1f}%", 
                                        'Healthy': f"{preds[2]*100:.1f}%"
                                    },
                                    'description': disease_info.get('description', ''),
                                    'remedies': disease_info.get('remedies', []),
                                    'prevention': disease_info.get('prevention', '')
                                })
    
    return render_template('index.html', result=None)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    print("Starting server...")
    print(f"Model input shape: {model.input_shape}")
    app.run(debug=True)
    