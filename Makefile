
# Minimal Makefile: assumes ImGui is installed system-wide and enables it
CC = g++
# enable ImGui by default; project expects a system-installed ImGui
# IMGUI is always compiled in-source now so we don't need the macro here
CFLAGS = -std=c++17 -O2 -g

# Use pkg-config for GLFW and Vulkan includes; also add common ImGui/stb include paths
INCLUDES = `pkg-config --cflags glfw3 vulkan` -I/usr/include/imgui -I/usr/include/stb
LIBS = `pkg-config --libs glfw3 vulkan` -limgui -lstb -ljpeg

SRC = main.cpp vulkan/*cpp
OUT = app

all:
	$(CC) $(CFLAGS) $(INCLUDES) $(SRC) -o $(OUT) $(LIBS)

clean:
	rm -f $(OUT)

