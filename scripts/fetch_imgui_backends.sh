#!/bin/bash
# Download ImGui backends (GLFW and Vulkan) for local build
set -e
IMGUI_VERSION=1.90.4
IMGUI_DIR="third_party/imgui"
BACKENDS=(imgui_impl_glfw.h imgui_impl_glfw.cpp imgui_impl_vulkan.h imgui_impl_vulkan.cpp)

# Ensure backends directory exists
mkdir -p "$IMGUI_DIR/backends"

# Download each backend file from the official ImGui repository
for file in "${BACKENDS[@]}"; do
    url="https://raw.githubusercontent.com/ocornut/imgui/v$IMGUI_VERSION/backends/$file"
    dest="$IMGUI_DIR/backends/$file"
    echo "Downloading $file ..."
    curl -fsSL "$url" -o "$dest"
done

echo "ImGui backends downloaded to $IMGUI_DIR/backends/"
