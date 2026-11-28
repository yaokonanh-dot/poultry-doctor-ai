
import os
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

app = FastAPI(title="Poultry Doctor AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")


@app.get("/")
def home():
    return {"message": "Poultry Doctor AI API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(
    photo: UploadFile = File(...),
    photo_type: str = Form("fiente"),
    bird_age: str = Form("unknown"),
    flock_size: str = Form("unknown"),
    affected_count: str = Form("unknown"),
    symptoms: str = Form(""),
    duration: str = Form("unknown"),
    notes: str = Form(""),
):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="AI service is not configured yet."
        )

    image_bytes = await photo.read()

    if len(image_bytes) > 8 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="Photo must be smaller than 8 MB."
        )

    if not photo.content_type or not photo.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:{photo.content_type};base64,{image_base64}"

    prompt = f"""
You are Poultry Doctor AI, a cautious poultry-health assistant
helping farmers in Africa.

Analyze the provided poultry image together with this information.

Photo type: {photo_type}
Bird age: {bird_age}
Flock size: {flock_size}
Number affected: {affected_count}
Symptoms: {symptoms}
Duration: {duration}
Additional notes: {notes}

Respond in clear, simple French using these sections:

1. OBSERVATIONS: What is visibly observable.
2. HYPOTHÈSES: Possible explanations, not confirmed diagnoses.
3. NIVEAU DE RISQUE: Low, medium, or high, with explanation.
4. ACTIONS IMMÉDIATES: Safe practical actions.
5. ALERTE: Signs requiring urgent veterinary assistance.
6. LIMITES: Explain what cannot be confirmed from a photo.

Do not claim a confirmed diagnosis from a photo.
Do not prescribe antibiotics or medication dosages.
If blood, severe weakness, breathing difficulty, rapid spread,
or unusual mortality is reported, recommend urgent veterinary help.
"""

    try:
        response = client.responses.create(
            model=MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": image_url,
                            "detail": "high"
                        }
                    ]
                }
            ]
        )

        return {
            "success": True,
            "analysis": response.output_text
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Analysis failed. Please try again later."
        )
