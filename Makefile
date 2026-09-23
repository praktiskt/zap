.PHONY: build run smoke test bench build-all clean docker-build docker-run docker-tag docker-push helm-lint

build:
	cargo build --release

run: build
	cargo run --release -- serve --bind 127.0.0.1:3279

smoke: build
	cargo run --release --example smoke

test:
	cargo test

bench:
	cargo bench --bench search -- --noplot queries

build-all:
	@mkdir -p dist
	cargo build --release --target x86_64-unknown-linux-musl  --target-dir dist/tgt-amd64 && mv dist/tgt-amd64/x86_64-unknown-linux-musl/release/zap dist/zap-linux-amd64
	cargo build --release --target aarch64-unknown-linux-musl --target-dir dist/tgt-arm64 && mv dist/tgt-arm64/aarch64-unknown-linux-musl/release/zap dist/zap-linux-arm64
	cargo build --release --target x86_64-apple-darwin    --target-dir dist/tgt-mac-amd64 && mv dist/tgt-mac-amd64/x86_64-apple-darwin/release/zap dist/zap-darwin-amd64
	cargo build --release --target aarch64-apple-darwin  --target-dir dist/tgt-mac-arm64 && mv dist/tgt-mac-arm64/aarch64-apple-darwin/release/zap dist/zap-darwin-arm64
	rm -rf dist/tgt-*
	sha256sum dist/* | sed "s/dist\///gi"
	@echo "note: cross targets need: rustup target add x86_64-unknown-linux-musl aarch64-unknown-linux-musl x86_64-apple-darwin aarch64-apple-darwin (+ macOS SDK / cargo-zigbuild for darwin on linux)"

clean:
	cargo clean
	rm -rf dist zap

docker-build:
	docker build -t zap:build .

docker-run: docker-build
	docker run --rm -p 3279:3279 zap:build serve --bind 0.0.0.0:3279

docker-tag: docker-build
	docker tag zap:build praktiskt/zap:latest

docker-push: docker-tag
	docker push praktiskt/zap:latest

helm-lint:
	helm lint charts/zap
	helm template charts/zap
