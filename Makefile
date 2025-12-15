
# Minimal Makefile: assumes ImGui is installed system-wide and enables it
CC = g++
# Build configuration: choose release or debug
# Usage: make            # builds default (release)
#        make BUILD=debug
# or shortcuts: make debug  or make release
BUILD ?= release
ifeq ($(BUILD),debug)
	CFLAGS = -std=c++20 -O0 -g -DDEBUG
else
	CFLAGS = -std=c++20 -O3 -DNDEBUG
endif

# Use pkg-config for GLFW and Vulkan includes; also add common ImGui/stb include paths
INCLUDES = `pkg-config --cflags glfw3 vulkan` -I/usr/include/imgui -I/usr/include/stb
LIBS = `pkg-config --libs glfw3 vulkan` -limgui -lstb -ljpeg -lgdal -lz

# Output directory for runtime binary and resources
OUT_DIR = bin

# shader sources and generated SPIR-V
SRCS := $(wildcard main.cpp utils/*.cpp vulkan/*.cpp widgets/*.cpp events/*.cpp math/*.cpp sdf/*.cpp space/*.cpp)
# Exclude legacy utils Camera implementation (migrated to math/Camera)
SRCS := $(filter-out utils/Camera.cpp,$(SRCS))
OBJ_DIR := $(OUT_DIR)/obj
OBJS := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(SRCS))
OUT = $(OUT_DIR)/app

# Objects used for the standalone server (exclude the app's main.o and vulkan objects to avoid duplicate main
# and linking against Vulkan)
SERVER_OBJS := $(filter-out $(OBJ_DIR)/main.o $(OBJ_DIR)/vulkan/%.o $(OBJ_DIR)/widgets/%.o $(OBJ_DIR)/events/KeyboardPublisher.o $(OBJ_DIR)/events/GamepadPublisher.o,$(OBJS))

# Server-specific link flags: exclude glfw and vulkan libs (they are brought in via pkg-config in LIBS)
SERVER_LIBS := -lstb -lgdal -lz
SERVER_INCLUDES := -I/usr/include/imgui -I/usr/include/stb

# shader sources and generated SPIR-V

SHADERS = shaders/main.vert shaders/main.tesc shaders/main.tese shaders/main.frag shaders/shadow.frag shaders/perlin_noise.comp shaders/sky.vert shaders/sky.frag
# SPIR-V targets will be written into the OUT_DIR/shaders directory
SPVS = $(SHADERS:.vert=.vert.spv)
SPVS := $(SPVS:.frag=.frag.spv)
SPVS := $(SPVS:.comp=.comp.spv)
SPVS := $(SPVS:.tesc=.tesc.spv)
SPVS := $(SPVS:.tese=.tese.spv)
OUT_SPVS := $(patsubst shaders/%, $(OUT_DIR)/shaders/%, $(SPVS))

all: shaders $(OUT) server
	@mkdir -p $(OUT_DIR)
	@mkdir -p $(OBJ_DIR)

.PHONY: server
server:
	@mkdir -p $(OUT_DIR)
	# Ensure project objects are built, then link server with them so it can use project symbols
	@$(MAKE) $(SERVER_OBJS)
	$(CC) $(CFLAGS) $(SERVER_INCLUDES) server.cpp $(SERVER_OBJS) -o $(OUT_DIR)/server $(SERVER_LIBS)

$(OUT): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) $(OBJS) -o $(OUT) $(LIBS)

	@echo "Copying runtime resources to $(OUT_DIR)/"
	@mkdir -p $(OUT_DIR)/shaders
	@if [ -d textures ]; then cp -a textures $(OUT_DIR)/ || true; fi
	@if [ -f imgui.ini ]; then cp imgui.ini $(OUT_DIR)/ || true; fi

# Pattern rule: compile each .cpp into an object under $(OBJ_DIR), preserving subdirs
$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	@echo "Compiling: $<"
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@


shaders: $(OUT_SPVS)
	@# Copy compiled SPIR-V back to the source shaders/ folder so FileReader can load shaders/*.spv at runtime
	@mkdir -p shaders
	@cp -u $(OUT_DIR)/shaders/*.spv shaders/ 2>/dev/null || true

$(OUT_DIR)/shaders/%.vert.spv: shaders/%.vert
	@echo "Compiling shader: $< -> $@"
	@mkdir -p $(dir $@)
	@if command -v glslc >/dev/null 2>&1; then \
		glslc -I shaders/includes $< -o $@; \
	else \
		glslangValidator -I shaders/includes -V $< -o $@; \
	fi

$(OUT_DIR)/shaders/%.frag.spv: shaders/%.frag
	@echo "Compiling shader: $< -> $@"
	@mkdir -p $(dir $@)
	@if command -v glslc >/dev/null 2>&1; then \
		glslc -I shaders/includes $< -o $@; \
	else \
		glslangValidator -I shaders/includes -V $< -o $@; \
	fi

$(OUT_DIR)/shaders/%.comp.spv: shaders/%.comp
	@echo "Compiling shader: $< -> $@"
	@mkdir -p $(dir $@)
	@if command -v glslc >/dev/null 2>&1; then \
		glslc -I shaders/includes $< -o $@; \
	else \
		glslangValidator -I shaders/includes -V $< -o $@; \
	fi

$(OUT_DIR)/shaders/%.tesc.spv: shaders/%.tesc
	@echo "Compiling shader: $< -> $@"
	@mkdir -p $(dir $@)
	@if command -v glslc >/dev/null 2>&1; then \
		glslc -I shaders/includes $< -o $@; \
	else \
		glslangValidator -I shaders/includes -V $< -o $@; \
	fi

$(OUT_DIR)/shaders/%.tese.spv: shaders/%.tese
	@echo "Compiling shader: $< -> $@"
	@mkdir -p $(dir $@)
	@if command -v glslc >/dev/null 2>&1; then \
		glslc -I shaders/includes $< -o $@; \
	else \
		glslangValidator -I shaders/includes -V $< -o $@; \
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
	sudo apt-get install robin-map-dev
	
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