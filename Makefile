.PHONY: debug release run run-debug clean
MAKE_JOBS ?= 8

# Minimal Makefile: assumes ImGui is installed system-wide and enables it
CC = g++
# Build configuration: choose release or debug
# Usage: make            # builds default (release)
#        make BUILD=debug
# or shortcuts: make debug  or make release
BUILD ?= release
ifeq ($(BUILD),debug)
	CFLAGS = -std=c++23 -O0 -g -DDEBUG -pthread -Ithird_party/imgui

else
	CFLAGS = -std=c++23 -O3 -march=native -DNDEBUG -pthread -DUSE_IMGUI -Ithird_party/imgui
endif

# Use pkg-config for GLFW and Vulkan includes; also add common ImGui/stb include paths
INCLUDES = `pkg-config --cflags glfw3 vulkan` -Ithird_party/imgui -Ithird_party/imgui/backends -I/usr/include/stb
LIBS = `pkg-config --libs glfw3 vulkan` -lstb -ljpeg -lgdal -lz

# Output directory for runtime binary and resources
OUT_DIR = bin

# Output directory for runtime binary and resources
OUT_DIR = bin
OBJ_DIR := $(OUT_DIR)/obj
IMGUI_CORE_SRCS := third_party/imgui/imgui.cpp third_party/imgui/imgui_draw.cpp third_party/imgui/imgui_tables.cpp third_party/imgui/imgui_widgets.cpp third_party/imgui/imgui_demo.cpp
IMGUI_BACKEND_SRCS := third_party/imgui/backends/imgui_impl_vulkan.cpp third_party/imgui/backends/imgui_impl_glfw.cpp
IMGUI_SRCS := $(IMGUI_CORE_SRCS) $(IMGUI_BACKEND_SRCS)
IMGUI_CORE_OBJS := $(patsubst third_party/imgui/%.cpp,$(OBJ_DIR)/imgui/%.o,$(IMGUI_CORE_SRCS))
IMGUI_BACKEND_OBJS := $(patsubst third_party/imgui/backends/%.cpp,$(OBJ_DIR)/imgui/backends/%.o,$(IMGUI_BACKEND_SRCS))
IMGUI_OBJS := $(IMGUI_CORE_OBJS) $(IMGUI_BACKEND_OBJS)
# shader sources and generated SPIR-V
SRCS := $(wildcard main.cpp utils/*.cpp vulkan/*.cpp widgets/*.cpp events/*.cpp math/*.cpp sdf/*.cpp space/*.cpp)
# Exclude legacy utils Camera implementation (migrated to math/Camera)
SRCS := $(filter-out utils/Camera.cpp,$(SRCS))
OBJ_DIR := $(OUT_DIR)/obj

# Compose object lists, then forcibly filter out any absolute /imgui/*.o
OBJS := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(SRCS)) $(IMGUI_OBJS)

OUT = $(OUT_DIR)/app

# Objects used for the standalone server (exclude the app's main.o and vulkan objects to avoid duplicate main
# and linking against Vulkan)
SERVER_OBJS := $(filter-out $(OBJ_DIR)/main.o $(OBJ_DIR)/vulkan/%.o $(OBJ_DIR)/widgets/%.o $(OBJ_DIR)/events/KeyboardPublisher.o $(OBJ_DIR)/events/GamepadPublisher.o,$(OBJS))

# Server-specific link flags: now include glfw and vulkan libs for ImGui backends
SERVER_LIBS := $(LIBS)
SERVER_INCLUDES := -Ithird_party/imgui -Ithird_party/imgui/backends -I/usr/include/stb


# Automatically find all shader source files in shaders/ with known extensions
SHADER_EXTS = vert frag geom comp tesc tese
SHADERS = $(foreach ext,$(SHADER_EXTS),$(wildcard shaders/*.$(ext)))
# Map each shader to its corresponding .spv output in bin/shaders, preserving extension
OUT_SPVS = \
	$(patsubst shaders/%.vert, $(OUT_DIR)/shaders/%.vert.spv, $(wildcard shaders/*.vert)) \
	$(patsubst shaders/%.frag, $(OUT_DIR)/shaders/%.frag.spv, $(wildcard shaders/*.frag)) \
	$(patsubst shaders/%.geom, $(OUT_DIR)/shaders/%.geom.spv, $(wildcard shaders/*.geom)) \
	$(patsubst shaders/%.comp, $(OUT_DIR)/shaders/%.comp.spv, $(wildcard shaders/*.comp)) \
	$(patsubst shaders/%.tesc, $(OUT_DIR)/shaders/%.tesc.spv, $(wildcard shaders/*.tesc)) \
	$(patsubst shaders/%.tese, $(OUT_DIR)/shaders/%.tese.spv, $(wildcard shaders/*.tese))


# Recursively create all object directories needed for all sources
define make-obj-dirs
	@mkdir -p $(OUT_DIR)
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(OBJ_DIR)/imgui
	@mkdir -p $(OBJ_DIR)/imgui/backends
	@find utils vulkan widgets events math sdf space -type d 2>/dev/null | while read dir; do \
		mkdir -p $(OBJ_DIR)/$$dir; \
	done
endef

all: imgui shaders $(OUT) server
	$(call make-obj-dirs)
	@mkdir -p $(OBJ_DIR)/imgui

.PHONY: imgui
imgui: $(IMGUI_OBJS)
	@echo "ImGui compilation complete"

.PHONY: server
server:
	@mkdir -p $(OUT_DIR)
	@# Ensure project objects are built, then link server with them so it can use project symbols
	@$(MAKE) -s --no-print-directory $(SERVER_OBJS)
	@$(CC) $(CFLAGS) $(SERVER_INCLUDES) server.cpp $(SERVER_OBJS) -o $(OUT_DIR)/server $(SERVER_LIBS)

$(OUT): $(OBJS)
	@echo "Linking: $(OUT)"
	@$(CC) $(CFLAGS) $(INCLUDES) $(OBJS) -o $(OUT) $(LIBS)

	@echo "Copying runtime resources to $(OUT_DIR)/"
	@mkdir -p $(OUT_DIR)/shaders
	@if [ -d textures ]; then cp -a textures $(OUT_DIR)/ || true; fi
	@if [ -f imgui.ini ]; then cp imgui.ini $(OUT_DIR)/ || true; fi

# Pattern rule: compile each .cpp into an object under $(OBJ_DIR), preserving subdirs

# Pattern rule for normal sources
$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	@echo "Compiling: $<"
	@$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@


$(OBJ_DIR)/imgui/%.o: third_party/imgui/%.cpp
	@mkdir -p $(OBJ_DIR)/imgui
	@echo "Compiling ImGui: $<"
	@$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/imgui/backends/%.o: third_party/imgui/backends/%.cpp
	@mkdir -p $(OBJ_DIR)/imgui/backends
	@echo "Compiling ImGui Backend: $<"
	@$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@


shaders: $(OUT_SPVS)
	@# Copy compiled SPIR-V back to the source shaders/ folder so FileReader can load shaders/*.spv at runtime
	@mkdir -p shaders
	@cp -u $(OUT_DIR)/shaders/*.spv shaders/ 2>/dev/null || true


# Generic pattern rule for all shader extensions in $(SHADER_EXTS)
define SHADER_COMPILE_RULE
$(OUT_DIR)/shaders/%.$(1).spv: shaders/%.$(1)
	@echo "Compiling shader: $$< -> $$@"
	@mkdir -p $$(dir $$@)
	@if command -v glslc >/dev/null 2>&1; then \
		glslc -I shaders/includes $$< -o $$@; \
	else \
		glslangValidator -I shaders/includes -V --target-env vulkan1.1 $$< -o $$@; \
	fi
endef

$(foreach ext,$(SHADER_EXTS),$(eval $(call SHADER_COMPILE_RULE,$(ext))))

.PHONY: debug release


release:
	@$(MAKE) --no-print-directory BUILD=release all

.PHONY: run run-debug
run: all
	@echo "Running app from $(OUT_DIR)/"
	@cd $(OUT_DIR) && ./app

run-debug: debug
	@echo "Running debug build from $(OUT_DIR)/"
	@cd $(OUT_DIR) && ./app

clean:
	@if [ "$(RESET)" = "1" ]; then reset; fi
	# Remove bin/ directory
	rm -rf $(OUT_DIR)
	# Remove generated SPIR-V files in shaders/ (if present)
	-rm -f $(SPVS)
	
debug:
	@$(MAKE) --no-print-directory -j$(MAKE_JOBS) BUILD=debug all
	
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

	@echo "Optionally, for profiling and code analysis tools, run:"
	@echo "  sudo apt-get install cloc kcachegrind massif-visualizer"

.PHONY: cloc
cloc:
	@echo "Running cloc to count lines of code..."
	@# Exclude runtime bins and third_party from the count; print to terminal (no file)
	@if command -v cloc >/dev/null 2>&1; then \
		cloc --exclude-dir=$(OUT_DIR),third_party .; \
	else \
		echo "cloc not found on PATH. Install it (e.g. sudo apt install cloc) to get a detailed LOC report."; \
	fi

callgrind:
	cd bin; valgrind --tool=callgrind --callgrind-out-file=file.out ./server;  kcachegrind file.out