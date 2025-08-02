import io
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
import segmentation_models_pytorch as smp
import tifffile
import torch.nn.functional as F

# === Configuration ===
MODEL_PATH = r"D:\Cellula_tech_intern\Week5\best_unet_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SIZE = (256, 256)
IN_CHANNELS = 128  # match your training input channel count

# === Load the trained model ===
def load_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # Required since input is not RGB
        in_channels=IN_CHANNELS,
        classes=1,
        activation="sigmoid"
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# === FastAPI App ===
app = FastAPI(title="Segmentation U-Net API", description="Upload a 128-channel image (e.g. .tiff or .npy) to receive a predicted mask")

# === Preprocess uploaded image ===
def preprocess_multichannel_image(image_bytes: bytes, ext: str) -> torch.Tensor:
    if ext.endswith('.npy'):
        image = np.load(io.BytesIO(image_bytes))
    elif ext.endswith('.tif') or ext.endswith('.tiff'):
        image = tifffile.imread(io.BytesIO(image_bytes))
    else:
        raise ValueError("Only .tif, .tiff, or .npy images with 128 channels are supported.")
    
    if image.ndim == 2:
        image = image[np.newaxis, :, :]
    elif image.ndim == 3 and image.shape[0] != IN_CHANNELS:
        if image.shape[-1] == IN_CHANNELS:
            image = np.transpose(image, (2, 0, 1))  # (C, H, W)
        else:
            raise ValueError(f"Image must have {IN_CHANNELS} channels. Got shape: {image.shape}")

    image = image.astype(np.float32)
    if image.max() > 1:
        image /= 255.0

    image_tensor = torch.tensor(image)
    image_tensor = F.interpolate(
        image_tensor.unsqueeze(0), size=TARGET_SIZE, mode='bilinear', align_corners=False
    ).squeeze(0)

    # Standardize
    image_tensor = (image_tensor - image_tensor.mean()) / (image_tensor.std() + 1e-8)
    return image_tensor.unsqueeze(0).to(DEVICE)  # (1, C, H, W)

# === Prediction ===
def predict_mask(image_tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        pred = model(image_tensor)
        mask = (pred.squeeze(0).squeeze(0) > 0.5).cpu().numpy().astype(np.uint8) * 255
    return mask

# === API Endpoint ===
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    ext = file.filename.lower()
    try:
        image_bytes = await file.read()
        input_tensor = preprocess_multichannel_image(image_bytes, ext)
        mask = predict_mask(input_tensor)

        # Return PNG image
        mask_img = Image.fromarray(mask)
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
