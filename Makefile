
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

# Output directory for runtime binary and resources
OUT_DIR = bin

# shader sources and generated SPIR-V
SRC = main.cpp vulkan/*cpp widgets/*cpp events/*cpp
OUT = $(OUT_DIR)/app

# shader sources and generated SPIR-V

SHADERS = shaders/triangle.vert shaders/triangle.tesc shaders/triangle.tese shaders/triangle.frag shaders/shadow.vert shaders/shadow.tesc shaders/shadow.tese shaders/shadow.frag shaders/perlin_noise.comp
# SPIR-V targets will be written into the OUT_DIR/shaders directory
SPVS = $(SHADERS:.vert=.vert.spv)
SPVS := $(SPVS:.frag=.frag.spv)
SPVS := $(SPVS:.comp=.comp.spv)
SPVS := $(SPVS:.tesc=.tesc.spv)
SPVS := $(SPVS:.tese=.tese.spv)
OUT_SPVS := $(patsubst shaders/%, $(OUT_DIR)/shaders/%, $(SPVS))

all: shaders
	@mkdir -p $(OUT_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(SRC) -o $(OUT) $(LIBS)

	# Copy runtime resources into bin/
	@echo "Copying runtime resources to $(OUT_DIR)/"
	@mkdir -p $(OUT_DIR)/shaders
	@if [ -d textures ]; then cp -a textures $(OUT_DIR)/ || true; fi
	@if [ -f imgui.ini ]; then cp imgui.ini $(OUT_DIR)/ || true; fi

shaders: $(OUT_SPVS)

$(OUT_DIR)/shaders/%.vert.spv: shaders/%.vert
	@echo "Compiling shader: $< -> $@"
	@mkdir -p $(dir $@)
	@if command -v glslc >/dev/null 2>&1; then \
		glslc $< -o $@; \
	else \
		glslangValidator -V $< -o $@; \
	fi

$(OUT_DIR)/shaders/%.frag.spv: shaders/%.frag
	@echo "Compiling shader: $< -> $@"
	@mkdir -p $(dir $@)
	@if command -v glslc >/dev/null 2>&1; then \
		glslc $< -o $@; \
	else \
		glslangValidator -V $< -o $@; \
	fi

$(OUT_DIR)/shaders/%.comp.spv: shaders/%.comp
	@echo "Compiling shader: $< -> $@"
	@mkdir -p $(dir $@)
	@if command -v glslc >/dev/null 2>&1; then \
		glslc $< -o $@; \
	else \
		glslangValidator -V $< -o $@; \
	fi

$(OUT_DIR)/shaders/%.tesc.spv: shaders/%.tesc
	@echo "Compiling shader: $< -> $@"
	@mkdir -p $(dir $@)
	@if command -v glslc >/dev/null 2>&1; then \
		glslc $< -o $@; \
	else \
		glslangValidator -V $< -o $@; \
	fi

$(OUT_DIR)/shaders/%.tese.spv: shaders/%.tese
	@echo "Compiling shader: $< -> $@"
	@mkdir -p $(dir $@)
	@if command -v glslc >/dev/null 2>&1; then \
		glslc $< -o $@; \
	else \
		glslangValidator -V $< -o $@; \
	fi

.PHONY: debug release
debug:
	@$(MAKE) BUILD=debug all

release:
	@$(MAKE) BUILD=release all

.PHONY: run run-debug
run: all
	@echo "Running app from $(OUT_DIR)/"
	@cd $(OUT_DIR) && ./app

run-debug: debug
	@echo "Running debug build from $(OUT_DIR)/"
	@cd $(OUT_DIR) && ./app

clean:
	# Remove built executable and bin/ runtime bundle
	rm -f $(OUT)
	rm -rf $(OUT_DIR)
	# Remove generated SPIR-V files in shaders/ (if present)
	-rm -f $(SPVS)
	
install:
	sudo apt install vulkan-validationlayers
	sudo apt install glslang-tools

	# 1. Install dependencies
	sudo apt update
	sudo apt install -y build-essential git cmake pkg-config libglfw3-dev libvulkan-dev vulkan-validationlayers glslang-tools

	# 2. Clone Dear ImGui
	mkdir -p third_party
	cd third_party
	git clone https://github.com/ocornut/imgui.git
	cd imgui

	# 3. Compile ImGui core + backends
	g++ -std=c++17 -O2 -fPIC -I. -Ibackends $(pkg-config --cflags glfw3 vulkan) \
		-c imgui.cpp imgui_draw.cpp imgui_tables.cpp imgui_widgets.cpp imgui_demo.cpp \
		backends/imgui_impl_glfw.cpp backends/imgui_impl_vulkan.cpp

	# 4. Create static library
	ar rcs libimgui.a *.o

	# 5. Install system-wide
	sudo mkdir -p /usr/local/include/imgui/backends
	sudo cp *.h /usr/local/include/imgui/
	sudo cp backends/*.h /usr/local/include/imgui/backends/
	sudo cp libimgui.a /usr/local/lib/
	sudo ldconfig

.PHONY: cloc
cloc:
	@echo "Running cloc to count lines of code..."
	@# Exclude runtime bins and third_party from the count; print to terminal (no file)
	@if command -v cloc >/dev/null 2>&1; then \
		cloc --exclude-dir=$(OUT_DIR),third_party .; \
	else \
		echo "cloc not found on PATH. Install it (e.g. sudo apt install cloc) to get a detailed LOC report."; \
	fi