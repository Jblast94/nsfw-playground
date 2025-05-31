from fastapi import FastAPI, HTTPException
from transformers import pipeline

app = FastAPI()
generator = pipeline('text-generation', model='gpt-neo-1.3B', device=-1)  # CPU mode

@app.post("/generate-text")
async def generate_text(prompt: str, max_length: int = 100, api_key: str = None):
    if api_key != "YOUR_SECRET_KEY":
        raise HTTPException(status_code=403, detail="Invalid API key")
    try:
        result = generator(prompt, max_length=max_length)[0]['generated_text']
        return {"status": "success", "text": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-image")
async def generate_image(prompt: str, api_key: str = None):
    if api_key != "YOUR_SECRET_KEY":
        raise HTTPException(status_code=403, detail="Invalid API key")
    try:
        # Placeholder for image generation logic
        # In a real scenario, you would integrate with an image generation model/API here
        dummy_image_url = "https://via.placeholder.com/500x300?text=Generated+Image+Placeholder"
        return {"status": "success", "image_url": dummy_image_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))