import os
import io
import tempfile
import subprocess

import edge_tts
import imageio_ffmpeg

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from google import genai
from google.genai import types

app = FastAPI(title="ESP32 Gemini Voice Assistant")

client = None


@app.on_event("startup")
async def startup():
    global client
    key = os.getenv("GEMINI_API_KEY")
    if key:
        client = genai.Client(api_key=key)


@app.get("/")
def home():
    return {"status": "Server running"}


@app.post("/voice")
async def voice(audio: UploadFile = File(...)):
    if client is None:
        raise HTTPException(500, "GEMINI_API_KEY is missing")

    data = await audio.read()
    if not data:
        raise HTTPException(400, "No audio received")

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                "Answer the user's spoken question briefly and naturally.",
                types.Part.from_bytes(
                    data=data,
                    mime_type="audio/wav"
                )
            ]
        )

        reply = response.text
        if not reply:
            raise HTTPException(502, "Gemini returned no text")

        with tempfile.TemporaryDirectory() as folder:
            mp3 = os.path.join(folder, "reply.mp3")
            await edge_tts.Communicate(
                reply,
                voice="en-US-AriaNeural"
            ).save(mp3)

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            result = subprocess.run(
                [
                    ffmpeg, "-y", "-i", mp3,
                    "-f", "wav", "-acodec", "pcm_s16le",
                    "-ar", "16000", "-ac", "1", "pipe:1"
                ],
                capture_output=True,
                check=True
            )

        return StreamingResponse(
            io.BytesIO(result.stdout),
            media_type="audio/wav"
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Processing failed: {exc}")
