# Reproducible environment for glsolver (CPU only).
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/glsolver
COPY pyproject.toml CMakeLists.txt README.md LICENSE ./
COPY src ./src
COPY reference ./reference
COPY tests ./tests
COPY benchmarks ./benchmarks
COPY examples ./examples
COPY visualization ./visualization
COPY scripts ./scripts
COPY docs ./docs
COPY regression ./regression

RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir ".[dev]"

# Smoke test: build succeeded and the core imports.
RUN python -c "import glsolver, glsolver._core; print('glsolver', glsolver.__version__, 'core', glsolver._core.__version__)"

ENTRYPOINT ["glsolve"]
CMD ["--help"]
