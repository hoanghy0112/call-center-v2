from io import BytesIO
from urllib.request import urlopen
import librosa
import torch
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
    TextStreamer,
)
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

streamer = TextStreamer(tokenizer)


# @timing_decorator
# def inference(conversation):
#     return "Hello, I'm an AI assistant made by Hy, what can I help you? A rainbow is a meteorological phenomenon that is caused by reflection, refraction and dispersion of light in water droplets resulting in a spectrum of light appearing in the sky. In your production system, you probably have a frontend created with a modern framework like React, Vue.js or Angular. And to communicate using WebSockets with your backend you would probably use your frontend's utilities. Or you might have a native mobile application that communicates with your WebSocket backend directly, in native code."
#     # return "Hello, I'm an AI assistant made by Hy, what can I help you? "


@timing_decorator
def inference(conversation):
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
    inputs.input_ids = inputs.input_ids.to("cuda:0")
    input_ids = inputs.input_ids.to("cuda:0")
    attention_mask = inputs.attention_mask.to("cuda:0")
    input_features = inputs.input_features.to("cuda:0")
    feature_attention_mask = inputs.feature_attention_mask.to("cuda:0")

    generate_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_features=input_features,
        feature_attention_mask=feature_attention_mask,
        max_length=1024,
        streamer=streamer,
    )
    # generate_ids = generate_ids[:, inputs.input_ids.size(1) :]

    response  = []
    for chunk in streamer: 
        print(chunk)
        response.append(chunk)

    # response = processor.batch_decode(
    #     generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )[0]

    return response.join(' ')
