# Bruce — reproducible build environment.
#
# Build:   docker build -t bruce:0.1 .
# Test:    docker run --rm bruce:0.1 cargo test -p bruce-core --release
# Demo:    docker run --rm bruce:0.1 bruce demo
# Python:  docker run --rm bruce:0.1 python -c "import bruce; print(bruce.__version__)"
#
# Reviewer-grade reproducibility: a single docker build reproduces the
# entire toolchain, all unit tests pass, and the CLI demo runs end-to-end.

FROM rust:1.95-slim-bookworm AS build

# minimal system deps (no GPU drivers here; we keep the image lean)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config libssl-dev python3 python3-pip python3-dev \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# install maturin for the Python wheel
RUN python3 -m pip install --break-system-packages --no-cache-dir maturin~=1.5 numpy

WORKDIR /bruce
COPY Cargo.toml Cargo.lock* ./
COPY bruce-core   ./bruce-core
COPY bruce-py     ./bruce-py
COPY bruce-cli    ./bruce-cli
COPY bruce-server ./bruce-server

# build Rust crates + Python wheel + HTTP server
RUN cargo build -p bruce-cli    --release \
    && cargo build -p bruce-server --release \
    && cargo test  -p bruce-core   --release \
    && cd bruce-py && maturin build --release \
    && python3 -m pip install --break-system-packages /bruce/target/wheels/bruce-*.whl

# install CLI + server binaries at /usr/local/bin
RUN cp /bruce/target/release/bruce        /usr/local/bin/bruce \
    && cp /bruce/target/release/bruce-server /usr/local/bin/bruce-server \
    && chmod +x /usr/local/bin/bruce /usr/local/bin/bruce-server

# smoke-test the assembled image as the LAST build step so a broken
# build fails the docker build itself
RUN bruce demo \
    && python3 -c "import bruce, numpy as np; \
                     op = bruce.Operator(eps=1.0, sim='dot'); \
                     out = op.attention(np.array([1., 0.]), np.array([[1., 0.], [0., 1.]]), np.array([[10., 0.], [0., 10.]])); \
                     assert abs(out[0] - 10*2.718281828/(2.718281828+1)) < 1e-10; \
                     print('Bruce wheel ok')"

# ----------- final runtime image ----------
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-numpy ca-certificates curl python3-pip \
    && rm -rf /var/lib/apt/lists/*

# copy artefacts only — no rust toolchain in the runtime image
COPY --from=build /usr/local/bin/bruce        /usr/local/bin/bruce
COPY --from=build /usr/local/bin/bruce-server /usr/local/bin/bruce-server
COPY --from=build /bruce/target/wheels /tmp/wheels
RUN python3 -m pip install --break-system-packages --no-cache-dir /tmp/wheels/bruce-*.whl \
    && rm -rf /tmp/wheels

# Run as a non-root user (industrial container hygiene: a container
# escape or RCE in the server lands in an unprivileged account).
RUN useradd --system --create-home --uid 10001 bruce
USER bruce
WORKDIR /home/bruce

# Container-level liveness probe against the server's /health
# (only meaningful when the container runs bruce-server).
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health || exit 1

EXPOSE 8080
CMD ["bruce", "demo"]
