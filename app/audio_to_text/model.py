import torch
import time
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from threading import Thread

from app.utils.elapsed_decorator import timing_decorator

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct",
    device_map="auto",
    quantization_config=quantization_config,
)

tokenizer = processor.tokenizer
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@timing_decorator
def inference(conversation):
    start_time = time.time()

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(ele["audio"])

    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    inputs.input_ids = inputs.input_ids.to(device)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    input_features = inputs.input_features.to(device)
    feature_attention_mask = inputs.feature_attention_mask.to(device)

    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_features=input_features,
        feature_attention_mask=feature_attention_mask,
        max_length=4024,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)

    thread.start()

    end_time = None

    response = ""
    for chunk in streamer:
        if end_time == None:
            end_time = time.time()
            print("Time to first bytes: ", end_time - start_time)

        response += chunk

        splitArray = response.split(".")
        if len(splitArray) > 1:
            response = ".".join(splitArray[1:])
            yield splitArray[0]
