from api import F5TTS
import random
import sys
from importlib.resources import files

import soundfile as sf
import tqdm
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    transcribe,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)
from f5_tts.model.utils import seed_everything



if __name__ == "__main__":
    f5tts = F5TTS()

    wav, sr, spec = f5tts.infer(
        ref_file=str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav")),
        ref_text="some call me nature, others call me mother nature.",
        gen_text="""I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring. Respect me and I'll nurture you; ignore me and you shall face the consequences.""",
        file_wave=str(files("f5_tts").joinpath("../../tests/api_out.wav")),
        file_spec=str(files("f5_tts").joinpath("../../tests/api_out.png")),
        seed=None,
    )

    print("seed :", f5tts.seed)