FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WIREOPE_CONF=/opt/wireope/configs

WORKDIR /opt/wireope

COPY requirements.txt requirements-dev.txt pyproject.toml README.md LICENSE ./
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

COPY configs ./configs
COPY src ./src
COPY scripts ./scripts
RUN python -m pip install --no-deps -e .

ENTRYPOINT ["python", "-m", "wireope.cli.verify"]
CMD ["--repo-root", "/opt/wireope"]
