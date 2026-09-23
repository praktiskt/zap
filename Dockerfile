FROM rust:1-alpine AS build
WORKDIR /app
RUN apk add --no-cache musl-dev
COPY Cargo.toml Cargo.lock ./
COPY benches benches
COPY src /app/src
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo build --release --bin zap && cp target/release/zap /zap

FROM scratch
COPY --from=build /zap /zap
USER 65534
ENTRYPOINT [ "/zap" ]
