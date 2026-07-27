FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SAMPLE_STEPS=4 \
    NUM_SAMPLES=1 \
    MERGE_STRATEGY=mean \
    SEED=0

WORKDIR /app

# PyTorch and NumPy are provided by the base image; NiBabel is the only
# additional runtime dependency.
RUN python -m pip install --no-cache-dir nibabel==5.3.2

# Only inference-time RARF code is copied into the submission image.
COPY rarf/__init__.py rarf/region_aware_flows.py /app/rarf/
COPY rarf/models /app/rarf/models
COPY brats/docker/inference.py /app/inference.py

ARG MODEL_CHECKPOINT
COPY ${MODEL_CHECKPOINT} /app/model.ckpt

CMD ["python", "/app/inference.py"]
