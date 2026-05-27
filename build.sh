#!/usr/bin/env bash
# Build script for cross-platform binaries
# Usage: ./build.sh
# Output: ./build/ (deepseek-cursor-proxy-{os}-{arch})

set -euo pipefail

BUILD_DIR="build"
APP_NAME="deepseek-cursor-proxy"
LDFLAGS="-s -w"

PLATFORMS=(
    "windows/amd64"
    "windows/arm64"
    "linux/amd64"
    "linux/arm64"
    "darwin/amd64"
    "darwin/arm64"
)

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

for PLATFORM in "${PLATFORMS[@]}"; do
    GOOS="${PLATFORM%/*}"
    GOARCH="${PLATFORM#*/}"
    EXT=""
    [[ "$GOOS" == "windows" ]] && EXT=".exe"

    BINARY="$APP_NAME-$GOOS-$GOARCH$EXT"
    echo "Building $BINARY ..."

    CGO_ENABLED=0 GOOS="$GOOS" GOARCH="$GOARCH" \
        go build -ldflags "$LDFLAGS" -o "$BUILD_DIR/$BINARY" ./cmd/deepseek-cursor-proxy/
done

echo ""
echo "Done. Binaries in ./$BUILD_DIR/:"
ls -1 "$BUILD_DIR"
