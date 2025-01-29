from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the trained model
model = tf.keras.models.load_model("diabetic_retinopathy_model.h5")

# Define class labels (modify if needed)
class_labels = [
    "No Retinopathy (Healthy Eye)",
    "Mild Non-Proliferative Diabetic Retinopathy (NPDR)",
    "Moderate Non-Proliferative Diabetic Retinopathy (NPDR)",
    "Severe Non-Proliferative Diabetic Retinopathy (NPDR)",
    "Proliferative Diabetic Retinopathy (PDR)"
]

# Create FastAPI instance
app = FastAPI(title="Diabetic Retinopathy Detection API")

# Image Preprocessing Function
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model input
    image = np.array(image) / 255.0   # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make a prediction
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])  # Get highest probability class

        # Prepare response
        response = {
            "filename": file.filename,
            "prediction": class_labels[predicted_class],
            "confidence": float(np.max(predictions[0]))
        }
        return response

    except Exception as e:
        return {"error": str(e)}

# Run the API with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
