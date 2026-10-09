# Build a partir da raiz do repo:  docker build -f deploy/Dockerfile -t go2-server:local .
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
WORKDIR /srv
# cuBLAS e cuDNN vêm do extra `gpu`; sem GPU no container o Whisper usa a CPU.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    HF_HOME=/models/huggingface \
    NVIDIA=/srv/.venv/lib/python3.12/site-packages/nvidia
ENV LD_LIBRARY_PATH=$NVIDIA/cublas/lib:$NVIDIA/cudnn/lib

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --extra gpu

COPY app ./app
COPY catalog.yaml ./

EXPOSE 9000
HEALTHCHECK --interval=30s --timeout=3s --start-period=120s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9000/health', timeout=2)"
CMD ["uv", "run", "--frozen", "--no-dev", "--extra", "gpu", "uvicorn", "app.main:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "9000"]
