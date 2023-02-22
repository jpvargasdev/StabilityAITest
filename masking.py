import os
import io
import warnings

from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

import numpy as np

from torchvision.transforms import GaussianBlur

os.environ["STABILITY_HOST"] = "grpc.stability.ai:443"
os.environ["STABILITY_KEY"] =  "STABILITY_KEY"


stability_api = client.StabilityInference(
        host=os.environ["STABILITY_HOST"],
        key=os.environ["STABILITY_KEY"],
        verbose=True,
        engine="stable-diffusion-v1-5",
    )

answers=stability_api.generate(
        prompt="a cat with dog ears, swiming in the space on old paper",
        seed=3123,
        steps=30,
    )

for resp in answers:
    for artifact in resp.artifacts:
        if artifact.finish_reason == generation.FILTER:
            warnings.warn(
                "Your request activated the API's safety filters and could not be processed."
                "Please modify the prompt and try again."
            )
            
        if artifact.type == generation.ARTIFACT_IMAGE:
            global img
            img = Image.open(io.BytesIO(artifact.binary))
            img.save(str(artifact.seed)+ "-1-start.png")


