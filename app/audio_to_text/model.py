# from io import BytesIO
# from urllib.request import urlopen
# import librosa
# import torch
# from transformers import (
#     Qwen2AudioForConditionalGeneration,
#     AutoProcessor,
#     BitsAndBytesConfig,
# )

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )

# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
# model = Qwen2AudioForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-Audio-7B-Instruct",
#     device_map="auto",
#     quantization_config=quantization_config,
# )


def inference(conversation):
    return "Hello, how are you?"


# def inference(conversation):
#     text = processor.apply_chat_template(
#         conversation, add_generation_prompt=True, tokenize=False
#     )
#     audios = []
#     for message in conversation:
#         if isinstance(message["content"], list):
#             for ele in message["content"]:
#                 if ele["type"] == "audio":
#                     audios.append(ele["audio"])

#     inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
#     inputs.input_ids = inputs.input_ids.to("cuda:0")
#     input_ids = inputs.input_ids.to("cuda:0")
#     attention_mask = inputs.attention_mask.to("cuda:0")
#     input_features = inputs.input_features.to("cuda:0")
#     feature_attention_mask = inputs.feature_attention_mask.to("cuda:0")

#     generate_ids = model.generate(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         input_features=input_features,
#         feature_attention_mask=feature_attention_mask,
#         max_length=1024,
#     )
#     generate_ids = generate_ids[:, inputs.input_ids.size(1) :]

#     response = processor.batch_decode(
#         generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )[0]

#     return response


conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav",
            },
        ],
    },
    {"role": "assistant", "content": "Yes, the speaker is female and in her twenties."},
]
