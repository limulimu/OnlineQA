from beam import Image, endpoint

image = (
    Image(base_image="nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04")
    .add_commands([        
        "apt-get update",
        "apt-get install git ffmpeg -y",
        "git clone https://github.com/limulimu/OnlineQA.git /tts"
    ])
    .add_python_packages(["f5-tts", "fastapi[standard]"])

)

@endpoint(image=image,gpu="RTX4090")
def handler():
    from f5_tts.api import F5TTS
    from fastapi.responses import FileResponse
    
    f5tts = F5TTS()
    wav, sr, spec = f5tts.infer(
        ref_file="/tts/1.wav",
        ref_text="张小明早上骑着白马飞过桥，看见一群绿鸭子在水中游，忽然听到天空中飞机轰鸣，对面的小孩说，九月的月亮真亮",
        gen_text="你是不是李永强",
        file_wave="/tts/lyq.wav",
        seed=None,
    )
    return FileResponse(
        path="/tts/lyq.wav",
        media_type="audio/wav",
        filename="audio.wav"
    )

