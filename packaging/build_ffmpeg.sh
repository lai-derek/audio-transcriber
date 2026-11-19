#!/usr/bin/env bash
# Run in an MSYS2 UCRT64 shell after prepare_ffmpeg.py.
set -euo pipefail
cd "$(dirname "$0")/.."
repo_dir="$PWD"
export PATH="/ucrt64/bin:/usr/bin:$PATH"
mkdir -p build/ffmpeg-compile build/ffmpeg
cd build/ffmpeg-compile
"$repo_dir/build/ffmpeg-source/ffmpeg-9.0.2/configure" \
  --target-os=mingw32 --arch=x86_64 --cc=gcc --disable-autodetect \
  --disable-gpl --disable-nonfree --disable-network --disable-x86asm \
  --disable-doc --disable-debug --disable-shared --enable-static --disable-everything \
  --enable-ffmpeg --disable-ffplay --disable-ffprobe \
  --enable-protocol=file --enable-demuxer=wav,mp3,mov \
  --enable-decoder=mp3,mp3float,aac,alac,pcm_s16le,pcm_s24le,pcm_s32le,pcm_f32le \
  --enable-parser=aac,mpegaudio --enable-filter=aresample,aformat,anull \
  --enable-muxer=wav --enable-encoder=pcm_s16le --extra-ldflags=-static
make -j"${NUMBER_OF_PROCESSORS:-4}"
cp ffmpeg.exe "$repo_dir/build/ffmpeg/ffmpeg.exe"
./ffmpeg.exe -version > "$repo_dir/build/ffmpeg/build-info.txt"
