#!/usr/bin/env bash
# Runs the command; when running as root (needed for ncu), hands files written to
# /workspace/.cache back to the host user afterwards.
set -u
mkdir -p "${HOME:-/tmp/home}" 2>/dev/null || true
"$@"
status=$?
if [ "$(id -u)" = 0 ] && [ -n "${HOST_UID:-}" ] && [ -d /workspace/.cache ]; then
    chown -R "${HOST_UID}:${HOST_GID:-$HOST_UID}" /workspace/.cache 2>/dev/null || true
fi
exit $status
