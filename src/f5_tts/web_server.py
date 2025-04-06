
import torch,torchaudio
from torch.quantization import QuantStub, DeQuantStub
from importlib.resources import files

from pydantic import BaseModel
from cached_path import cached_path

import soundfile as sf
from infer.utils_infer import infer_batch_process, preprocess_ref_audio_text, load_vocoder, load_model
from model.backbones.dit import DiT
from fastapi import FastAPI


app = FastAPI()
# vocab_file=r"/root/F5-TTS/src/f5_tts/infer/examples/vocab.txt"
# ref_audio=r"/root//F5-TTS/src/f5_tts/6.mp3"
vocab_file=r"f:\F5-TTS\src\f5_tts\infer\examples\vocab.txt"
ref_audio=r"f:\F5-TTS\src\f5_tts\6.mp3"
ref_text="猕猴桃满满一颗全是果肉"
dtype=torch.float16
ckpt_file= str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"))
# device = ( "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device = "cpu"
# Load the model using the provided checkpoint and vocab files
model = load_model(
    model_cls=DiT,
    model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
    ckpt_path=ckpt_file,
    mel_spec_type="vocos",  # or "bigvgan" depending on vocoder
    vocab_file=vocab_file,
    ode_method="euler",
    use_ema=True,
    device=device,
).to(device, dtype=dtype)

# model.qconfig = torch.quantization.get_default_qconfig('fbgemm') 
# torch.quantization.prepare(model, inplace=True)
device = torch.device("cuda")
model.to(device)

# Load the vocoder
vocoder = load_vocoder(is_local=False)

# Set sampling rate for streaming
sampling_rate = 24000  # Consistency with client


class GenItem(BaseModel):
    message: str
    id: int



# @app.on_event("startup")
# async def startup():
#     """Warm up the model with a dummy input to ensure it's ready for real-time processing."""
#     print("Warming up the model...")
#     ref_audio, ref_text = preprocess_ref_audio_text(ref_audio, ref_text)
#     audio, sr = torchaudio.load(ref_audio)
#     gen_text = "Warm-up text for the model."

#     # Pass the vocoder as an argument here
#     infer_batch_process((audio, sr), ref_text, [gen_text], model, vocoder, device=device)
#     print("Warm-up completed.")

@app.post("/gen")
def generate(item:GenItem):
    r_audio, r_text = preprocess_ref_audio_text(ref_audio, ref_text)

    # # Load reference audio
    audio, sr = torchaudio.load(r_audio)

    # Run inference for the input text
    audio_chunk, final_sample_rate, _ = infer_batch_process(
        (audio, sr),
        r_text,
        [item.message],
        model,
        vocoder,
        device=device,  # Pass vocoder here
    )
    sf.write("here.wav", audio_chunk, final_sample_rate)