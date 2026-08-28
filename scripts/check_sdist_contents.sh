#!/usr/bin/env bash
#
# Fail if the source distribution contains files that must never be
# published (issue #264).
#
# Background: with a git file-finder in the build environment setuptools
# swept every tracked path into the sdist, including `figures/` — ~17.6 MB
# of rendered PNGs, two of them 8.4 MB each. MANIFEST.in now declares the
# sdist contents explicitly; this script is the gate that keeps it honest,
# so a future MANIFEST.in edit (or a build backend that starts guessing
# again) fails the build instead of quietly re-inflating every release.
#
# Usage:  scripts/check_sdist_contents.sh [dist-dir]      (default: dist)
#
# Exits non-zero, listing the offending members, if the sdist contains any
# forbidden path. Also prints the archive size so the number is visible in
# the CI log.

set -euo pipefail

DIST_DIR="${1:-dist}"

shopt -s nullglob
sdists=("${DIST_DIR}"/*.tar.gz)
shopt -u nullglob

if [ "${#sdists[@]}" -eq 0 ]; then
    echo "sdist guard: no *.tar.gz found in '${DIST_DIR}/' — did the build run?" >&2
    exit 1
fi
if [ "${#sdists[@]}" -gt 1 ]; then
    echo "sdist guard: expected exactly one sdist in '${DIST_DIR}/', found ${#sdists[@]}:" >&2
    printf '  %s\n' "${sdists[@]}" >&2
    exit 1
fi

SDIST="${sdists[0]}"

# Paths that must not ship. Matched against the member path with the
# leading "<name>-<version>/" prefix stripped, so the patterns read the
# same as the repository layout.
#   figures/            rendered validation figures (the 17.6 MB trap)
#   validation/*.png|csv  regenerated driver outputs
#   docs/_build/        built HTML docs
#   .github/ .claude/   repo-only tooling, useless to a downstream user
#   __pycache__/        build-host byte-code
FORBIDDEN_RE='^(figures/|docs/_build/|\.github/|\.claude/|.*__pycache__/|validation/[^/]*\.(png|csv)$)'

members="$(tar -tzf "${SDIST}" | sed -E 's|^[^/]*/||')"
offenders="$(printf '%s\n' "${members}" | grep -E "${FORBIDDEN_RE}" || true)"

size_bytes="$(wc -c < "${SDIST}" | tr -d '[:space:]')"
echo "sdist guard: ${SDIST} — $(printf '%s' "${members}" | grep -c . ) members, ${size_bytes} bytes"

if [ -n "${offenders}" ]; then
    echo "" >&2
    echo "ERROR: the sdist contains paths that must never be published:" >&2
    printf '  %s\n' ${offenders} >&2
    echo "" >&2
    echo "These are rendered outputs or repo-only tooling. Fix MANIFEST.in" >&2
    echo "(prune/recursive-exclude the path) rather than relaxing this guard;" >&2
    echo "shipping figures/ alone inflated the sdist by ~17.6 MB." >&2
    exit 1
fi

echo "sdist guard: OK — no forbidden paths."
