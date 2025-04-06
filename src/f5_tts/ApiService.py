from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import soundfile as sf
import os
from fastapi import FastAPI, File, UploadFile,HTTPException
from fastapi.responses import FileResponse
from qdrant_client import QdrantClient
from qdrant_client.models import Filter
import shutil
import requests
from qdrant_client.http.models import  PointStruct
import logging
import uuid
import librosa
import re
import cn2an
from importlib.resources import files
from pydantic import BaseModel
from cached_path import cached_path
import soundfile as sf
from fastapi import FastAPI
import platform
from api import F5TTS
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



app = FastAPI()


f5tts = F5TTS()


# client = QdrantClient(path="vectors")
client = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
collection_name = "qa_collection"

# Detect the operating system
os_name = platform.system()

# Set directory paths based on OS
if os_name == "Linux":
    BASE_DIR = "/root/files"
    GEN_DIR = "/root/files"
elif os_name == "Windows":
    BASE_DIR = "./"
    GEN_DIR = "./"
else:
    # Default fallback (you can modify this as needed)
    BASE_DIR = "./"
    GEN_DIR = "./"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")  # 设置时间格式
logger = logging.getLogger("uvicorn")



class GenItem(BaseModel):
    message: str
    id: int

# @app.on_event("startup")
# async def startup():
#     # 执行一次性初始化任务，如数据库连接等
#     global client
#     # 初始化 Qdrant 客户端
#     client = QdrantClient(path="vectors")

#     # 创建 Qdrant 集合
#     collection_name = "qa_collection"


#     # 加载中文嵌入模型
#     global model
#     model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')






def FileUpload(id):
    url = "http://localhost:8805/upload"
    file_path = "/root/files/{}.wav".format(str(id))

    with open(file_path, "rb") as f:
        files = {"file": f}
        try:
            response = requests.post(url, files=files, timeout=5)
            print(response.json())
        except:
            print("Upload failed")

def generate_uuid():
    my_uuid = uuid.uuid4()
    short_int = int(str(my_uuid.int)[:16])
    return short_int

# Define a request body model
class QueryRequest(BaseModel):
    question: str
    firm_id: int
    product_id: int
    sp_id: int = 3

class IndertRequest(BaseModel):
    question: str
    answer: str
    firm_id: int
    product_id: int
    sp_id: int = 3

class Item(BaseModel): 
    question: str 
    answer: str

class GenItem(BaseModel):
    message: str
    id: int
    sp_id: int = 3

@app.post("/qa")
async def get_similar_vector(item: QueryRequest):
    # 查询示例
    logger.info("starting similarity")
    query_question =  item.question
    query_embedding = model.encode([query_question])[0]
    logger.info("embedding")
    # 搜索最相似的问题
    if item.sp_id > 0:
        filter_conditions = Filter(
            must=[
                # 查询 category 为 "electronics" 的文档
                {"key": "firm_id", "match": {"value": item.firm_id}},
                # 查询 price 小于 1000 的文档
                {"key": "product_id", "match": {"value": item.product_id}},
                {"key": "sp_id", "match": {"value": item.sp_id}}
            ]
        )
    else:
        filter_conditions = Filter(
            must=[
                # 查询 category 为 "electronics" 的文档
                {"key": "firm_id", "match": {"value": item.firm_id}},
                # 查询 price 小于 1000 的文档
                {"key": "product_id", "match": {"value": item.product_id}}
            ]
        ) 

    search_results = client.search(
        collection_name="qa_collection",
        query_vector=query_embedding,
        limit=1,
        query_filter=filter_conditions,
        with_payload=True,
    )
    logger.info("starting search")
    # 输出最相似的问题及其答案
    # for result in search_results:
    if len(search_results)>0: 
        result= search_results[0]
        if result.score>0.8:
            # print(f"最相似的问题: {result.payload['question']}, 答案: {result.payload['answer']},距离:{result.score}")
            logger.info("done")
            return {"result": "success", "answer": result.payload['answer'],'url':'{}.wav'.format(str(result.payload['q_id']))}
        else:
            return {"result": "cannot answer"}
    else:
        return {"result": "cannot answer"}



@app.post("/add") 
async def create_item(item: IndertRequest): 
    """
        # API 的 URL
    url = 'http://139.84.136.212:6677/add'
    # 请求的头部
    headers = {
        'Content-Type': 'application/json'
    }

    # 请求的数据
    data = {
        question: str
        answer: str
        firm_id: int
        product_id: int
        sp_id: int
    }
        # 发送 POST 请求
    response = requests.post(url, headers=headers, json=data)

    # 输出响应
    print(response.status_code)
    print(response.json())
    """

    # New sentence to encode
    logger.info("adding item")
    new_embedding = model.encode([item.question])[0]

    # Connect to DuckDB
    filter_conditions = Filter(
        must=[
            # 查询 category 为 "electronics" 的文档
            {"key": "firm_id", "match": {"value": item.firm_id}},
            # 查询 price 小于 1000 的文档
            {"key": "product_id", "match": {"value": item.product_id}},
            {"key": "sp_id", "match": {"value": item.sp_id}}
        ]
    )
    search_results = client.search(
        collection_name="qa_collection",
        query_vector=new_embedding,
        limit=1,
        query_filter=filter_conditions,
        with_payload=True,
    )

    #check question exsit

    if len(search_results)>0 and item.question !="":
        logger.info("updating question")
        result= search_results[0]
        if result.score>0.95:
            print("Question already exists, updating")
            genitem=GenItem(id=result.payload['q_id'],message=item.answer, sp_id=item.sp_id) 
            GenVoice(genitem)
            return {'result':True,'url':'{}.wav'.format(str(result.payload['q_id']))}


    # cursor = client.count_points(collection_name="qa_collection")
    logger.info("Creating new")
    file_id = generate_uuid()
    genitem=GenItem(message=item.answer,id=file_id,sp_id=item.sp_id)
    GenVoice(genitem)
    
    # Insert new row into table
    if item.question!="":
        points = [
            PointStruct(id=file_id, vector=new_embedding, payload={"question": item.question, "answer": item.answer,"q_id":file_id,"firm_id":item.firm_id,"product_id":item.product_id,"sp_id":item.sp_id})
        ]
        client.upsert(collection_name="qa_collection", points=points)


    # con.close()
    return {'result':True,'url':'{}.wav'.format(str(file_id))}



@app.post("/gen")
def GenVoice(item:GenItem):
    """
        url = 'http://127.0.0.1:8000/gen'

    # 请求的头部
    headers = {
        'Content-Type': 'application/json'
    }

    # 请求的数据
    data = {
        'message': '说什么都可以的吧',
        'id': 1197032697424030,
        'sp_id': 1
    }


    # 发送 POST 请求
    response = requests.post(url, headers=headers, json=data)

    # 输出响应
    print(response.status_code)
    print(response.json())
    """
        # API 的 URL
    logger.info(item.message)
    # r_audio, r_text = preprocess_ref_audio_text(ref_audio.format(str(item.sp_id)), ref_text)
    requested_path = os.path.abspath(os.path.join(GEN_DIR, "{}.wav".format(str(item.id))))
    
    wav, sr, spec = f5tts.infer(
        ref_file=str(files("f5_tts").joinpath("infer/examples/basic/{}.wav".format(item.sp_id))),
        ref_text="张小明早上骑着白马飞过桥，看见一群绿鸭子在水中游，忽然听到天空中飞机轰鸣，对面的小孩说，九月的月亮真亮",
        gen_text=item.message,
        file_wave=requested_path,
        # file_spec=str(files("f5_tts").joinpath("../../tests/api_out.png")),
        seed=None,
    )

    # # # Load reference audio
    # audio, sr = torchaudio.load(r_audio)

    # # Run inference for the input text
    # audio_chunk, final_sample_rate, _ = infer_batch_process(
    #     (audio, sr),
    #     r_text,
    #     [item.message],
    #     model_gen,
    #     vocoder,
    #     device=device,  # Pass vocoder here
    # )
    # requested_path = os.path.abspath(os.path.join(GEN_DIR, "{}.wav".format(str(item.id))))
    # with open(requested_path, "wb") as f:
    #     sf.write(f.name, audio_chunk, final_sample_rate)
    #     logger.info(f"WAV file saved as {f.name}, size: {len(audio_chunk)} bytes")







@app.get("/files/{file_path:path}")
def get_file(file_path: str):
    """
        url = 'http://139.84.136.212:6677/files/368.wav'

    # 请求的头部
    headers = {
        'Content-Type': 'application/json'
    }



    # 发送 POST 请求
    response = requests.get(url, headers=headers)

    # 输出响应
    print(response.status_code)
    w=response.content
    with open('368.wav', 'wb') as f:
        f.write(w)
    """
    # 确保文件路径在允许的目录中
    requested_path = os.path.abspath(os.path.join(BASE_DIR, file_path))
    if not requested_path.startswith(BASE_DIR):
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not os.path.exists(requested_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    return FileResponse(requested_path, headers=headers)



def resample_wave(wavefile):
    # 读取音频文件
    audio, sr = sf.read(wavefile)  # sr 是原始采样率

    # 目标采样率
    target_sr = 16000

    # 使用 librosa 进行重采样
    resampled_audio = librosa.resample(audio.T, orig_sr=sr, target_sr=target_sr)

    # 保存重采样后的音频
    sf.write(wavefile, resampled_audio.T, target_sr)

    print("Resampling completed!")

def convert_arabic_to_chinese(text):
    # 使用正则表达式匹配文本中的阿拉伯数字
    def replace(match):
        number = match.group()
        return cn2an.an2cn(number)

    # 替换阿拉伯数字为汉字数字
    return re.sub(r'\d+', replace, text)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    print("uploading file")
    file_location = os.path.abspath(os.path.join(BASE_DIR, "{}".format(file.filename))) 
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"filename": file.filename, "file_location": file_location}


@app.get("/speed")
async def speed():
    return {"speed": 16000}