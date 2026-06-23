#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <input-compile-commands.json> <output-json> <clangd-db-dir>" >&2
  exit 2
fi

input_json=$1
output_json=$2
clangd_db_dir=$3

if [[ ! -f "$input_json" ]]; then
  echo "warning: $input_json does not exist; skipping clangd compilation database generation" >&2
  exit 0
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required to generate $output_json" >&2
  exit 1
fi

tmp_json="${output_json}.tmp"
mkdir -p "$(dirname "$output_json")" "$clangd_db_dir"

jq '
  def drop_generate_code:
    if type == "array" then
      map(select((type == "string" and startswith("--generate-code=arch")) | not))
    elif type == "string" then
      gsub("(^|[[:space:]])\"?--generate-code=arch=[^\"[:space:]]+\"?"; "")
    else
      .
    end;

  map(
    if has("arguments") then .arguments |= drop_generate_code else . end
    | if has("command") then .command |= drop_generate_code else . end
  )
' "$input_json" > "$tmp_json"
mv "$tmp_json" "$output_json"

ln -sfn "../$(basename "$output_json")" "$clangd_db_dir/compile_commands.json"
