import time
from app.core.config import settings

from app.utils.elapsed_decorator import timing_decorator

if settings.ENV == "gpu":
    from app.audio_to_text.model import inference
if settings.ENV == "ultravox":
    from app.audio_to_text.ultravox_model import inference
else:

    @timing_decorator
    def inference(conversation):
        text = "Hello, I'm an AI assistant made by Hee, what can I help you? A rainbow is a meteorological phenomenon that is caused by reflection, refraction and dispersion of light in water droplets resulting in a spectrum of light appearing in the sky. In your production system, you probably have a frontend created with a modern framework like React, Vue.js or Angular. And to communicate using WebSockets with your backend you would probably use your frontend's utilities. Or you might have a native mobile application that communicates with your WebSocket backend directly, in native code."

        for chunk in text.split("."):
            yield chunk
            time.sleep(0.5)