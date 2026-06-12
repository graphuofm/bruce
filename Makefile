# Bruce — one-command reproducibility.
#
#   make            — build everything (release)
#   make test       — cargo test all crates
#   make demo       — run the CLI demo
#   make wheel      — build the Python wheel
#   make python     — install the wheel and run a Python smoke test
#   make server     — build the bruce-server HTTP binary
#   make test-server — start bruce-server, hit every endpoint, kill
#   make docker     — build the all-in-one reproducible Docker image
#   make bench      — criterion microbenchmarks
#   make clean      — wipe target/

CARGO   ?= cargo
PYTHON  ?= python3
MATURIN ?= maturin
DOCKER  ?= docker

.PHONY: all test demo wheel python server test-server docker bench clean

all:
	$(CARGO) build --release

test:
	$(CARGO) test -p bruce-core   --release
	$(CARGO) test -p bruce-cli    --release
	$(CARGO) test -p bruce-server --release

demo:
	$(CARGO) run -p bruce-cli --release --quiet -- demo

wheel:
	cd bruce-py && $(MATURIN) build --release

python: wheel
	$(PYTHON) -m pip install --user --force-reinstall \
		--quiet target/wheels/bruce-*.whl
	$(PYTHON) -c "import bruce, numpy as np; \
		op = bruce.Operator(eps=1.0, sim='dot'); \
		out = op.attention(np.array([1., 0.]), \
		                   np.array([[1., 0.], [0., 1.]]), \
		                   np.array([[10., 0.], [0., 10.]])); \
		assert abs(out[0] - 10*2.718281828/(2.718281828+1)) < 1e-10; \
		Q = np.random.randn(32, 4); K = Q.copy(); V = Q.copy(); \
		p = bruce.chain_tree(32); \
		o = bruce.tree_attention(Q, K, V, p, eps=1.0); \
		assert o.shape == (32, 4); \
		print(f'Bruce {bruce.__version__} OK -> attention {out.tolist()}')"

server:
	$(CARGO) build -p bruce-server --release

test-server: server
	bash scripts/test_bruce_server.sh

docker:
	$(DOCKER) build -t bruce:0.1 .

bench:
	$(CARGO) bench -p bruce-core

clean:
	$(CARGO) clean
	rm -rf target/wheels
