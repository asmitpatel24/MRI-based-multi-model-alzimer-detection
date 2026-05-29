# ============================================
# Alzheimer Hybrid Model API — FastAPI + PyTorch
# ============================================
import uvicorn
import os
import traceback
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import io
import tempfile
from fpdf import FPDF
from datetime import datetime

from hybrid_model import AlzheimerHybridModel

# ============================================
# APP SETUP
# ============================================
app = FastAPI(
    title="Alzheimer MRI Classification API",
    description=(
        "Upload an MRI image and get the predicted Alzheimer stage. "
        "Uses a hybrid CNN + Vision Transformer + MLP fusion model."
    ),
    version="2.0",
)

# Enable CORS for frontend (Next.js localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# MODEL LOADING
# ============================================
MODEL_PATH = "alzheimer_hybrid_model.pth"
IMG_SIZE = 224
CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[*] Loading hybrid model...")
model = AlzheimerHybridModel(num_classes=4)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    print(f"[OK] Model loaded from {MODEL_PATH}")
else:
    print(f"[!] Warning: Model file '{MODEL_PATH}' not found.")
    print("[!] Using UNTRAINED model (random weights) for demonstration.")
    print(f"[!] Run 'python train.py' to train the model first.")

model.to(device)
model.eval()
print("[OK] Model loaded on", device)


# ============================================
# IMAGE PREPROCESSING
# ============================================
def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return TRANSFORM(img).unsqueeze(0).to(device)


# ============================================
# PREDICT ENDPOINT
# ============================================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    age: float = Form(70.0),
):
    """
    Predict Alzheimer stage from an MRI scan.

    Args:
        file: MRI image (JPEG/PNG)
        age:  Patient age (default 70)
    """
    try:
        image_bytes = await file.read()
        img_tensor = preprocess_image(image_bytes)

        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Log probabilities
            probs_list = probabilities[0].tolist()
            print(f"[DEBUG] Raw Logits: {outputs[0].tolist()}")
            print(f"[DEBUG] Probabilities: {dict(zip(CLASS_NAMES, [round(p, 4) for p in probs_list]))}")
            
            confidence_tensor, predicted_class_tensor = torch.max(probabilities, 1)

        predicted_label = CLASS_NAMES[predicted_class_tensor.item()]
        confidence_score = float(confidence_tensor.item()) * 100 # Convert to percentage
        
        print(f"[RESULT] Pred: {predicted_label}, Conf: {confidence_score:.4f}%")

        result = {
            "predicted_class": predicted_label,
            "confidence_pct": round(confidence_score, 2),
            "class_probabilities": {
                CLASS_NAMES[i]: round(float(probs_list[i]) * 100, 2)
                for i in range(len(CLASS_NAMES))
            },
            "model": "Hybrid CNN + ViT + MLP",
            "input_metadata": {"age": age},
        }

        if confidence_score < 60:
            result["warning"] = "Low confidence -- consider clinical review."

        return JSONResponse(content=result)

    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(content={"error": str(exc)}, status_code=500)


# ============================================
# GENERATE REPORT ENDPOINT
# ============================================
@app.post("/generate-report")
async def generate_report(
    file: UploadFile = File(...),
    predicted_class: str = Form(...),
    confidence_pct: float = Form(...),
    patient_name: str = Form("Not provided"),
    age: str = Form("Not specified"),
):
    """
    Generate a PDF medical report from the prediction.
    """
    try:
        # Read image
        image_bytes = await file.read()
        
        # Save image to temp file for FPDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img.save(temp_img, format="PNG")
            temp_img_path = temp_img.name

        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        
        # Hospital/Clinic Header
        pdf.set_font("Arial", 'B', 20)
        pdf.set_text_color(44, 62, 80) # Dark blue/gray
        pdf.cell(200, 10, txt="NEUROLOGY DEPARTMENT", ln=True, align='C')
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(52, 73, 94)
        pdf.cell(200, 8, txt="Automated AI Analysis Report", ln=True, align='C')
        
        pdf.ln(5)
        
        # Add a light gray line
        pdf.set_draw_color(189, 195, 199)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        # Patient Info
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(0, 0, 0)
        
        pdf.cell(50, 10, txt="Date:", border=0)
        pdf.set_font("Arial", '', 12)
        pdf.cell(100, 10, txt=datetime.now().strftime("%B %d, %Y - %H:%M"), border=0, ln=True)

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(50, 10, txt="Patient Name:", border=0)
        pdf.set_font("Arial", '', 12)
        pdf.cell(100, 10, txt=patient_name, border=0, ln=True)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(50, 10, txt="Patient Age:", border=0)
        pdf.set_font("Arial", '', 12)
        pdf.cell(100, 10, txt=str(age) + (" Years" if str(age).isdigit() else ""), border=0, ln=True)

        pdf.ln(10)

        # Result Banner
        is_high_risk = predicted_class != "NonDemented"
        
        pdf.set_font("Arial", 'B', 16)
        
        if predicted_class == "NonDemented":
            pdf.set_fill_color(223, 240, 216) # Light green
            pdf.set_text_color(60, 118, 61)   # Dark green
        else:
            pdf.set_fill_color(242, 222, 222) # Light red
            pdf.set_text_color(169, 68, 66)   # Dark red
            
        pdf.cell(0, 10, txt=f"   Prediction: {predicted_class}", ln=True, fill=True)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, txt=f"   Confidence Level: {confidence_pct:.2f}%", ln=True, fill=True)

        pdf.ln(5)

        # Image Section
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 8, txt="Analyzed MRI Scan:", ln=True)
        
        # Save current y
        current_y = pdf.get_y()
        # Add image, center it
        # Make the image 75x75 and adjust y offset to 5 below current_y
        pdf.image(temp_img_path, x=67.5, y=current_y + 2, w=75, h=75)
        
        # Move Y down below the image (75 + 2 + padding)
        pdf.set_y(current_y + 85)
        
        # Disclaimer
        pdf.set_font("Arial", 'B', 11)
        pdf.set_text_color(231, 76, 60) # Red
        pdf.cell(200, 8, txt="DISCLAIMER:", ln=True)
        
        pdf.set_font("Arial", 'I', 10)
        pdf.set_text_color(127, 140, 141) # Gray
        disclaimer_text = (
            "This report is generated by an Artificial Intelligence model (Hybrid CNN + Swin Transformer) "
            "and is highly experimental. It is NOT a definitive medical diagnosis. A certified medical "
            "professional or neurologist must conduct a formal evaluation before drawing any clinical conclusions."
        )
        pdf.multi_cell(0, 5, txt=disclaimer_text)

        # Save PDF to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            pdf_path = temp_pdf.name
        
        pdf.output(pdf_path)
        
        # Clean up the image
        try:
            os.remove(temp_img_path)
        except:
            pass

        return FileResponse(
            pdf_path, 
            media_type="application/pdf", 
            filename=f"Report_{predicted_class}.pdf",
            background=None # Note: FileResponse doesn't auto-delete by default without a background task unless returning a StreamingResponse, but for now we let the OS temp clear it or just leave it since it's tempfile.
        )

    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(content={"error": str(exc)}, status_code=500)


# ============================================
# ROOT ENDPOINT
# ============================================
@app.get("/")
def home():
    return {
        "message": "Alzheimer Hybrid Model API v2.0 -- POST /predict with an MRI image.",
        "model": "CNN + Vision Transformer + MLP Fusion",
        "classes": CLASS_NAMES,
    }


# ============================================
# RUN
# ============================================
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
