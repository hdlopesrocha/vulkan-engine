
# Minimal Makefile: assumes ImGui is installed system-wide and enables it
CC = g++
# Build configuration: choose release or debug
# Usage: make            # builds default (release)
#        make BUILD=debug
# or shortcuts: make debug  or make release
BUILD ?= release
ifeq ($(BUILD),debug)
	CFLAGS = -std=c++17 -O0 -g -DDEBUG
else
	CFLAGS = -std=c++17 -O3 -DNDEBUG
endif

# Use pkg-config for GLFW and Vulkan includes; also add common ImGui/stb include paths
INCLUDES = `pkg-config --cflags glfw3 vulkan` -I/usr/include/imgui -I/usr/include/stb
LIBS = `pkg-config --libs glfw3 vulkan` -limgui -lstb -ljpeg

SRC = main.cpp vulkan/*cpp
OUT = app

all:
	$(CC) $(CFLAGS) $(INCLUDES) $(SRC) -o $(OUT) $(LIBS)

.PHONY: debug release
debug:
	@$(MAKE) BUILD=debug all

release:
	@$(MAKE) BUILD=release all

clean:
	rm -f $(OUT)
	
install:
	sudo apt install vulkan-validationlayers
	sudo apt install glslang-tools

	# Install common build dependencies (Debian/Ubuntu)
	sudo apt update
	sudo apt install -y build-essential git cmake pkg-config libglfw3-dev libvulkan-dev

	# Install Dear ImGui if not available via pkg-config
	@if pkg-config --exists imgui; then \
		echo "imgui detected via pkg-config, skipping local install"; \
	else \
		echo "Installing Dear ImGui locally under /usr/local"; \
		mkdir -p third_party; \
		if [ ! -d third_party/imgui ]; then \
			git clone https://github.com/ocornut/imgui.git third_party/imgui; \
		fi; \
		pushd third_party/imgui > /dev/null; \
		# compile ImGui core + GLFW+Vulkan backends into a static library
		SRCS="imgui.cpp imgui_draw.cpp imgui_tables.cpp imgui_widgets.cpp imgui_demo.cpp backends/imgui_impl_glfw.cpp backends/imgui_impl_vulkan.cpp"; \
		CFLAGS_IMGUI="-std=c++17 -O2 -fPIC -I. -Ibackends `pkg-config --cflags glfw3 vulkan`"; \
		for f in $${SRCS}; do g++ $$CFLAGS_IMGUI -c $$f -o $${f//\//_}.o || exit 1; done; \
		ar rcs libimgui.a *.o; \
		sudo mkdir -p /usr/local/include/imgui; \
		sudo cp *.h /usr/local/include/imgui/; \
		sudo mkdir -p /usr/local/include/imgui/backends; \
		sudo cp backends/*.h /usr/local/include/imgui/backends/; \
		sudo cp libimgui.a /usr/local/lib/; \
		sudo ldconfig || true; \
		popd > /dev/null; \
	fi
