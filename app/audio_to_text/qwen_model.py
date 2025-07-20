import torch
import time
from transformers import (
    Qwen2AudioForConditionalGeneration,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
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

MODEL_NAME = "Qwen/Qwen2.5-Omni-3B"

processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_NAME)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_NAME,
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
    inputs = inputs.to(model.device).to(model.dtype)

    generation_kwargs = dict(
        **inputs,
        max_length=4024,
        streamer=streamer,
    )

    text_ids, audio = model.generate(**generation_kwargs)

    text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(text)
    sf.write(
        "output.wav",
        audio.reshape(-1).detach().cpu().numpy(),
        samplerate=24000,
    )

    return text

    # thread = Thread(target=model.generate, kwargs=generation_kwargs)

    # thread.start()

    # end_time = None

    # response = ""
    # for chunk in streamer:
    #     if end_time == None:
    #         end_time = time.time()
    #         print("Time to first bytes: ", end_time - start_time)

    #     response += chunk

    #     splitArray = response.split(".")
    #     if len(splitArray) > 1:
    #         response = ".".join(splitArray[1:])
    #         yield splitArray[0]
