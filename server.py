import sys
import base64
import tempfile
import soundfile
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from io import BytesIO
import torchaudio
from cosyvoice.utils.file_utils import load_wav
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

sys.path.append("third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice2


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.cosyvoice = CosyVoice2(
        "pretrained_models/CosyVoice2-0.5B", load_jit=False, load_trt=False, fp16=False
    )
    yield


app = FastAPI(lifespan=lifespan)


class TTSRequest(BaseModel):
    text: str
    prompt_audio: str  # Base64 encoded WAV
    speed: float


@app.post("/synthesize/sichuanese")
async def synthesize_sichuanese(request: TTSRequest, fastapi_request: Request):
    try:
        audio_bytes = base64.b64decode(request.prompt_audio)

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp.flush()

            _, sr = soundfile.read(tmp.name)
            prompt = load_wav(tmp.name, sr)

        cosyvoice = fastapi_request.app.state.cosyvoice
        outputs = []

        for segment in cosyvoice.inference_instruct2(
            request.text,
            "用四川话讲",
            prompt,
            stream=False,
            speed=request.speed,
        ):
            buffer = BytesIO()
            torchaudio.save(
                buffer, segment["tts_speech"], cosyvoice.sample_rate, format="wav"
            )
            buffer.seek(0)
            outputs.append(base64.b64encode(buffer.read()).decode("utf-8"))

        return {"audio": outputs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
