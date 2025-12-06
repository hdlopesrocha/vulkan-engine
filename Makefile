# Makefile
# ---------

CC = g++
CFLAGS = -std=c++17 -O2
INCLUDES = `pkg-config --cflags glfw3 vulkan`
LIBS = `pkg-config --libs glfw3 vulkan` -ljpeg

SRC = main.cpp
OUT = app

all:
	$(CC) $(CFLAGS) $(INCLUDES) $(SRC) -o $(OUT) $(LIBS)

clean:
	rm -f $(OUT)
	
install:
	sudo apt install vulkan-validationlayers
	sudo apt install glslang-tools
