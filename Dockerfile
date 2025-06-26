FROM r8.im/stability-ai/stable-diffusion-xl:latest

COPY predictor.py /src/
COPY mental_upgrade.safetensors /src/
