#!/usr/bin/env bash
# Push a version tag to trigger the Release workflow.
# Usage: ./tag.sh 1.0.0          → pushes tag v1.0.0
#        ./tag.sh 1.0.0 -m "msg" → annotated tag with message
set -euo pipefail

if [ $# -eq 0 ]; then
  echo "Usage: ./tag.sh <version> [-m <message>]"
  echo "Example: ./tag.sh 1.0.0"
  echo "         ./tag.sh 1.0.0 -m 'first stable release'"
  exit 1
fi

VERSION="$1"
TAG="v${VERSION#v}"
shift

# Check for uncommitted changes.
if ! git diff-index --quiet HEAD --; then
  echo "ERROR: working tree has uncommitted changes. Commit or stash them first."
  exit 1
fi

# Fetch tags so we don't accidentally reuse one.
git fetch --tags --quiet

if git rev-parse "$TAG" >/dev/null 2>&1; then
  echo "ERROR: tag $TAG already exists."
  exit 1
fi

if [ $# -gt 0 ]; then
  git tag -a "$TAG" "$@"
else
  git tag -a "$TAG" -m "$TAG"
fi

git push origin "$TAG"
echo ""
echo "Tag $TAG pushed. Release workflow:"
echo "  https://github.com/$(git remote get-url origin | sed 's|.*[:/]\(.*\)/\(.*\)\.git|\1/\2|')/actions/workflows/release.yml"
