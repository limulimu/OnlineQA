import modal
import torch

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git","ffmpeg")
    .pip_install("torch==2.6.0","f5-tts","fastapi[standard]")
    .env({"HALT_AND_CATCH_FIRE": "0"})
    .run_commands("git clone https://github.com/limulimu/OnlineQA.git /tts")
)

app = modal.App(name="tts",image=image)


@app.cls(gpu="L4")
class WebApp:
    @modal.enter()
    def startup(self):
        from f5_tts.api import F5TTS
        self.tts = F5TTS()

    @modal.fastapi_endpoint(method="get", docs=True)
    def goodbye(self):
        from fastapi.responses import FileResponse

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            wav, sr, spec = self.tts.infer(
                ref_file="/tts/1.wav",
                ref_text="张小明早上骑着白马飞过桥，看见一群绿鸭子在水中游，忽然听到天空中飞机轰鸣，对面的小孩说，九月的月亮真亮",
                gen_text="你是不是李永强",
                file_wave="/tts/lyq.wav",
                # file_spec=str(files("f5_tts").joinpath("../../tests/api_out.png")),
                seed=None,
            )
        return FileResponse(
                path="/tts/lyq.wav",
                media_type="audio/wav",  # Use "audio/wav" for WAV files, etc.
                filename="audio.wav"      # Optional: suggests a filename for download
        )
