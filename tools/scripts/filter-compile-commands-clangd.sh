#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <input-compile-commands.json> <output-json> <clangd-db-dir> <source-dir> <cuda-include-dirs>" >&2
  exit 2
fi

input_json=$1
output_json=$2
clangd_db_dir=$3
source_dir=$4
cuda_include_dirs=$5

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

cuda_isystem_flags=""
IFS=';' read -ra cuda_include_dir_array <<< "$cuda_include_dirs"
for cuda_include_dir in "${cuda_include_dir_array[@]}"; do
  if [[ -n "$cuda_include_dir" ]]; then
    cuda_isystem_flags+=" -isystem $(printf '%q' "$cuda_include_dir")"
  fi
done

find "$source_dir/include" -type f \( -name '*.h' -o -name '*.hpp' \) -print0 |
  jq -Rs --arg source_dir "$source_dir" --arg cuda_isystem_flags "$cuda_isystem_flags" '
    split("\u0000")
    | map(select(length > 0))
    | map({
        directory: $source_dir,
        command: ("/usr/bin/c++ -std=c++20 -I" + $source_dir + "/include" + $cuda_isystem_flags + " -x c++-header -c " + .),
        file: .
      })
  ' > "$headers_json"

jq --arg cuda_include_dirs "$cuda_include_dirs" '
  # nvcc spells the target GPU as --generate-code=arch=compute_120,code=[...],
  # which clang/clangd do not understand. Drop it and re-emit the equivalent
  # clang driver flag --cuda-gpu-arch=sm_120 so that .cu files are parsed for
  # the same architecture the real build uses (otherwise clangd falls back to
  # sm_52, where e.g. <crt/mma.hpp> never declares nvcuda::wmma).
  def gencode_archs:
    if type == "array" then
      map(select(type == "string")) | join(" ")
    elif type == "string" then
      .
    else
      ""
    end;

  # Highest arch wins, matching the ascending CUDA_ARCHITECTURES order.
  def last_gencode_arch:
    [gencode_archs
     | match("arch=compute_([0-9]+[a-z]?)"; "g")
     | .captures[0].string] | last;

  def translate_generate_code:
    if type == "array" then
      . as $arguments
      | ($arguments | last_gencode_arch) as $arch
      | ($arguments
         | map(select(
             if type == "string" then
               (test("^(--generate-code|-gencode)=arch") | not)
             else
               true
             end)))
        + (if $arch then ["--cuda-gpu-arch=sm_" + $arch] else [] end)
    elif type == "string" then
      . as $command
      | ($command | last_gencode_arch) as $arch
      | ($command
         | gsub("(^|[[:space:]])\"?(--generate-code|-gencode)=arch[^\"[:space:]]+\"?"; " "))
      | if $arch then . + " --cuda-gpu-arch=sm_" + $arch else . end
    else
      .
    end;

  def drop_clangd_unsupported_nvcc_flags:
    if type == "array" then
      map(select(. != "--expt-relaxed-constexpr"))
    elif type == "string" then
      gsub("(^|[[:space:]])--expt-relaxed-constexpr($|[[:space:]])"; " ")
    else
      .
    end;

  def cuda_isystem_args:
    ($cuda_include_dirs | split(";") | map(select(length > 0)) | map(["-isystem", .]) | add // []);

  def cuda_isystem_command:
    ($cuda_include_dirs | split(";") | map(select(length > 0)) | map(" -isystem " + @sh) | join(""));

  def is_cuda_command:
    if has("arguments") then
      (.arguments | any(. == "cu"))
    elif has("command") then
      (.command | test("(^|[[:space:]])-x[[:space:]]+cu([[:space:]]|$)"))
    else
      false
    end;

  map(
    if has("arguments") then .arguments |= translate_generate_code | .arguments |= drop_clangd_unsupported_nvcc_flags else . end
    | if has("command") then .command |= translate_generate_code | .command |= drop_clangd_unsupported_nvcc_flags else . end
    | if is_cuda_command and has("arguments") then .arguments += cuda_isystem_args else . end
    | if is_cuda_command and has("command") then .command += cuda_isystem_command else . end
  )
' "$input_json" |
  jq -s '.[0] + .[1]' - "$headers_json" > "$tmp_json"
mv "$tmp_json" "$output_json"
rm -f "$headers_json"

ln -sfn "../$(basename "$output_json")" "$clangd_db_dir/compile_commands.json"
