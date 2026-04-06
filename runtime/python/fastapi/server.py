# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import logging
import tempfile
import uuid
from datetime import datetime
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np


def _perf(event, **kv):
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    extra = " ".join(f"{k}={v}" for k, v in kv.items())
    print(f"[PERF] {ts} {event} {extra}", flush=True)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import AutoModel

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio


@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    model_output = cosyvoice.inference_sft(tts_text, spk_id)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    req_id = uuid.uuid4().hex[:8]
    _perf("COSYVOICE_REQUEST", req=req_id, mode="zero_shot", text_len=len(tts_text))
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(await prompt_wav.read())
        tmp_path = tmp.name
    def generate_and_cleanup():
        first_chunk_logged = False
        try:
            # 对短回复（<2s 语音），stream=True 会因为 model.py 中 100ms sleep 轮询
            # 反而变慢，故改回 stream=False。token_hop_len patch 保留但不生效。
            for chunk in generate_data(cosyvoice.inference_zero_shot(tts_text, prompt_text, tmp_path, stream=False)):
                if not first_chunk_logged:
                    _perf("COSYVOICE_FIRST_CHUNK", req=req_id)
                    first_chunk_logged = True
                yield chunk
            _perf("COSYVOICE_DONE", req=req_id)
        finally:
            os.unlink(tmp_path)
    return StreamingResponse(generate_and_cleanup())


@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(await prompt_wav.read())
        tmp_path = tmp.name
    def generate_and_cleanup():
        try:
            for chunk in generate_data(cosyvoice.inference_cross_lingual(tts_text, tmp_path)):
                yield chunk
        finally:
            os.unlink(tmp_path)
    return StreamingResponse(generate_and_cleanup())


@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(await prompt_wav.read())
        tmp_path = tmp.name
    def generate_and_cleanup():
        try:
            for chunk in generate_data(cosyvoice.inference_instruct2(tts_text, instruct_text, tmp_path)):
                yield chunk
        finally:
            os.unlink(tmp_path)
    return StreamingResponse(generate_and_cleanup())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='iic/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    parser.add_argument('--token_hop_len',
                        type=int,
                        default=10,
                        help='流式首段最小 token 数 (默认 25 对应 ~1s 语音，10 对应 ~0.4s，越小首段越快但音质略降)')
    args = parser.parse_args()
    cosyvoice = AutoModel(model_dir=args.model_dir)
    # 为了短回复场景的流式首段延迟，把 token_hop_len 从默认 25 调低到 10。
    # CosyVoice2 token rate 是 25Hz，token_hop_len=10 意味着首段生成 ~0.4s 语音即可 yield。
    if hasattr(cosyvoice.model, 'token_hop_len'):
        old_hop = cosyvoice.model.token_hop_len
        cosyvoice.model.token_hop_len = args.token_hop_len
        cosyvoice.model.token_max_hop_len = 4 * args.token_hop_len
        print(f"[CONFIG] token_hop_len: {old_hop} -> {args.token_hop_len} (token_max_hop_len -> {4 * args.token_hop_len})", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
