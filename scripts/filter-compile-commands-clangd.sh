#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <input-compile-commands.json> <output-json> <clangd-db-dir> <source-dir>" >&2
  exit 2
fi

input_json=$1
output_json=$2
clangd_db_dir=$3
source_dir=$4

if [[ ! -f "$input_json" ]]; then
  echo "warning: $input_json does not exist; skipping clangd compilation database generation" >&2
  exit 0
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required to generate $output_json" >&2
  exit 1
fi

tmp_json="${output_json}.tmp"
headers_json="${output_json}.headers.tmp"
mkdir -p "$(dirname "$output_json")" "$clangd_db_dir"

find "$source_dir/include" -type f \( -name '*.h' -o -name '*.hpp' \) -print0 |
  jq -Rs --arg source_dir "$source_dir" '
    split("\u0000")
    | map(select(length > 0))
    | map({
        directory: $source_dir,
        command: ("/usr/bin/c++ -std=c++20 -I" + $source_dir + "/include -x c++-header -c " + .),
        file: .
      })
  ' > "$headers_json"

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
' "$input_json" |
  jq -s '.[0] + .[1]' - "$headers_json" > "$tmp_json"
mv "$tmp_json" "$output_json"
rm -f "$headers_json"

ln -sfn "../$(basename "$output_json")" "$clangd_db_dir/compile_commands.json"
