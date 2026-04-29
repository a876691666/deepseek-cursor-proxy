# syntax=docker/dockerfile:1.6

# ---- Build stage ----
FROM golang:1.25-alpine AS build
WORKDIR /src

# Cache dependencies first for faster incremental builds.
COPY go.mod go.sum ./
RUN go mod download

# Copy source and build a statically-linked binary (modernc.org/sqlite is pure Go,
# so CGO is not required).
COPY . .
ENV CGO_ENABLED=0
RUN go build -trimpath -ldflags="-s -w" -o /out/deepseek-cursor-proxy ./cmd/deepseek-cursor-proxy

# ---- Runtime stage ----
FROM alpine:3.20
RUN apk add --no-cache ca-certificates tzdata && \
    addgroup -S app && adduser -S -G app app && \
    mkdir -p /home/app/.deepseek-cursor-proxy && \
    chown -R app:app /home/app

COPY --from=build /out/deepseek-cursor-proxy /usr/local/bin/deepseek-cursor-proxy

USER app
WORKDIR /home/app

EXPOSE 9000
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -qO- http://127.0.0.1:9000/healthz >/dev/null 2>&1 || exit 1

ENTRYPOINT ["/usr/local/bin/deepseek-cursor-proxy"]
CMD ["--host", "0.0.0.0", "--port", "9000"]
