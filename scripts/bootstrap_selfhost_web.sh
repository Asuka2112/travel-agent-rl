#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUNTIME_ROOT="$PROJECT_ROOT/.runtime"
SOURCE_ROOT="$RUNTIME_ROOT/sources"
TOOLCHAIN_ROOT="$RUNTIME_ROOT/toolchains"
DOWNLOAD_ROOT="$RUNTIME_ROOT/downloads"
NODE_ROOT="$TOOLCHAIN_ROOT/node-v22.14.0-linux-x64"
PNPM_ROOT="$TOOLCHAIN_ROOT/pnpm-9.15.9"
GO_ROOT="$TOOLCHAIN_ROOT/go1.24.1"
CARGO_ROOT="$TOOLCHAIN_ROOT/cargo"
RUSTUP_ROOT="$TOOLCHAIN_ROOT/rustup"

FIRECRAWL_REV=b57d5f2c4d600b2bf8b34507ee4c874c26c07bb0
SEARXNG_REV=374939b888c8644b408b793fe42d584454631cec
VALKEY_REV=d063dff523340e5a075f7ddc436e0c2a43d04061
NODE_SHA=69b09dba5c8dcb05c4e4273a4340db1005abeafe3927efda2bc5b249e80437ec
GO_SHA=cb2396bae64183cdccf81a9a6df0aea3bce9511fc21469fb89a0c00470088073
RUSTUP_SHA=4acc9acc76d5079515b46346a485974457b5a79893cfb01112423c89aeb5aa10

mkdir -p "$SOURCE_ROOT" "$TOOLCHAIN_ROOT" "$DOWNLOAD_ROOT" "$RUNTIME_ROOT/npm-cache"

checkout_fixed() {
  local repository=$1
  local revision=$2
  local destination=$3
  if [[ ! -d "$destination/.git" ]]; then
    git clone --filter=blob:none "$repository" "$destination"
  fi
  git -C "$destination" fetch --depth 1 origin "$revision"
  git -C "$destination" checkout --detach "$revision"
  test "$(git -C "$destination" rev-parse HEAD)" = "$revision"
}

checkout_fixed https://github.com/firecrawl/firecrawl.git "$FIRECRAWL_REV" \
  "$SOURCE_ROOT/firecrawl-v1.7.0"
checkout_fixed https://github.com/searxng/searxng.git "$SEARXNG_REV" \
  "$SOURCE_ROOT/searxng"
checkout_fixed https://github.com/valkey-io/valkey.git "$VALKEY_REV" \
  "$SOURCE_ROOT/valkey-8.1.3"

NODE_ARCHIVE="$DOWNLOAD_ROOT/node-v22.14.0-linux-x64.tar.xz"
if [[ ! -f "$NODE_ARCHIVE" ]]; then
  curl -fL --retry 5 -o "$NODE_ARCHIVE" \
    https://nodejs.org/dist/v22.14.0/node-v22.14.0-linux-x64.tar.xz
fi
printf '%s  %s\n' "$NODE_SHA" "$NODE_ARCHIVE" | sha256sum -c -
if [[ ! -x "$NODE_ROOT/bin/node" ]]; then
  tar -xJf "$NODE_ARCHIVE" -C "$TOOLCHAIN_ROOT"
fi
export PATH="$NODE_ROOT/bin:$PNPM_ROOT/bin:$PATH"
export npm_config_cache="$RUNTIME_ROOT/npm-cache"
if [[ ! -x "$PNPM_ROOT/bin/pnpm" ]]; then
  npm install --global --prefix "$PNPM_ROOT" pnpm@9.15.9
fi

GO_ARCHIVE="$DOWNLOAD_ROOT/go1.24.1.linux-amd64.tar.gz"
if [[ ! -f "$GO_ARCHIVE" ]]; then
  curl -fL --retry 5 -o "$GO_ARCHIVE" \
    https://go.dev/dl/go1.24.1.linux-amd64.tar.gz
fi
printf '%s  %s\n' "$GO_SHA" "$GO_ARCHIVE" | sha256sum -c -
if [[ ! -x "$GO_ROOT/bin/go" ]]; then
  tar -xzf "$GO_ARCHIVE" -C "$TOOLCHAIN_ROOT" --transform 's,^go,go1.24.1,'
fi

RUSTUP_INIT="$DOWNLOAD_ROOT/rustup-init"
if [[ ! -f "$RUSTUP_INIT" ]]; then
  curl -fL --retry 5 -o "$RUSTUP_INIT" \
    https://static.rust-lang.org/rustup/archive/1.28.2/x86_64-unknown-linux-gnu/rustup-init
fi
printf '%s  %s\n' "$RUSTUP_SHA" "$RUSTUP_INIT" | sha256sum -c -
chmod 700 "$RUSTUP_INIT"
if [[ ! -x "$CARGO_ROOT/bin/cargo" ]]; then
  CARGO_HOME="$CARGO_ROOT" RUSTUP_HOME="$RUSTUP_ROOT" RUSTUP_INIT_SKIP_PATH_CHECK=yes \
    "$RUSTUP_INIT" -y --no-modify-path --profile minimal --default-toolchain 1.85.1
fi

FIRECRAWL_ROOT="$SOURCE_ROOT/firecrawl-v1.7.0"
python3 "$PROJECT_ROOT/scripts/patch_firecrawl_playwright.py" \
  "$FIRECRAWL_ROOT/apps/playwright-service-ts/api.ts"

pushd "$FIRECRAWL_ROOT/apps/api" >/dev/null
pnpm install --frozen-lockfile
pnpm run build
popd >/dev/null

pushd "$FIRECRAWL_ROOT/apps/api/sharedLibs/go-html-to-md" >/dev/null
GOPROXY=https://goproxy.cn,direct "$GO_ROOT/bin/go" build \
  -buildmode=c-shared -o html-to-markdown.so html-to-markdown.go
popd >/dev/null

pushd "$FIRECRAWL_ROOT/apps/api/sharedLibs/html-transformer" >/dev/null
CARGO_HOME="$CARGO_ROOT" RUSTUP_HOME="$RUSTUP_ROOT" \
  "$CARGO_ROOT/bin/cargo" build --release
popd >/dev/null

pushd "$FIRECRAWL_ROOT/apps/playwright-service-ts" >/dev/null
npm ci --ignore-scripts
npm run build
PLAYWRIGHT_BROWSERS_PATH="$RUNTIME_ROOT/browsers" npx playwright install chromium
popd >/dev/null

make -C "$SOURCE_ROOT/valkey-8.1.3" -j"$(nproc)" BUILD_TLS=no MALLOC=libc

BOOTSTRAP_VENV="$RUNTIME_ROOT/venv-bootstrap"
SEARXNG_VENV="$RUNTIME_ROOT/venv-searxng311-runtime"
if [[ ! -x "$BOOTSTRAP_VENV/bin/python" ]]; then
  python3 -m venv "$BOOTSTRAP_VENV"
  "$BOOTSTRAP_VENV/bin/pip" install virtualenv==21.7.4
fi
if [[ ! -x "$SEARXNG_VENV/bin/python" ]]; then
  "$BOOTSTRAP_VENV/bin/python" -m virtualenv -p python3.11 "$SEARXNG_VENV"
fi
"$SEARXNG_VENV/bin/pip" install -e "$SOURCE_ROOT/searxng" \
  -r "$SOURCE_ROOT/searxng/requirements-server.txt"

python3 "$PROJECT_ROOT/scripts/manage_selfhost_web.py" start
