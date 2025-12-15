#!/usr/bin/env just --justfile
set shell := ["zsh", "-cu"]
set fallback

default:
  just -u --list

test:
  uv run pytest -v

lint:
  uv run ruff format --check .
  uv run ruff check .

typecheck:
  uv run pyright

dev:
  uv run pgsql-mcp --help

release-help:
  @echo "- update version in pyproject.toml"
  @echo "- uv sync"
  @echo "- git commit"
  @echo "- git push && merge to main"
  @echo '- just release 0.0.0 "note"'
  @echo 'OR'
  @echo '- just prerelease 0.0.0 1 "note"'

release version note extra="":
  #!/usr/bin/env bash
  if [[ "{{version}}" == v* ]]; then
    echo "Error: Do not include 'v' prefix in version. It will be added automatically."
    exit 1
  fi
  uv build && git tag -a "v{{version}}" -m "Release v{{version}}" || true && git push --tags && gh release create "v{{version}}" --title "pgsql-mcp v{{version}}" --notes "{{note}}" {{extra}} dist/*.whl dist/*.tar.gz

prerelease version rc note:
  just release "{{version}}rc{{rc}}" "{{note}}" "--prerelease"
