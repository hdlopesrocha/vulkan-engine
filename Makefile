.DEFAULT_GOAL := all
.PHONY: debug release run run-debug clean all _all imgui shaders server
MAKE_JOBS ?= 8

# Success/error jingles. The mp3s live in sounds/ and are played on build
# completion and after a run exits. The first available player wins; if none
# is installed the sounds are silently skipped. Playback is asynchronous
# (nohup + &) so the build/run never waits for the jingle to finish.
SOUND_DIR := $(CURDIR)/sounds
define PLAY_SOUND
{ f="$(SOUND_DIR)/$(1).mp3"; if [ -f "$$f" ]; then \
    if command -v mpv >/dev/null 2>&1; then nohup mpv --really-quiet --no-video "$$f" >/dev/null 2>&1 & \
    elif command -v ffplay >/dev/null 2>&1; then nohup ffplay -nodisp -autoexit -loglevel quiet "$$f" >/dev/null 2>&1 & \
    elif command -v mpg123 >/dev/null 2>&1; then nohup mpg123 -q "$$f" >/dev/null 2>&1 & \
    elif command -v cvlc >/dev/null 2>&1; then nohup cvlc --play-and-exit --intf dummy "$$f" >/dev/null 2>&1 & \
    fi; \
fi; }
endef

# Minimal Makefile: assumes ImGui is installed system-wide and enables it
CC = g++
# Build configuration: choose release or debug
# Usage: make            # builds default (release)
#        make BUILD=debug
# or shortcuts: make debug  or make release
BUILD ?= release
# CFLAGS is recursive so it re-evaluates $(BUILD) on every reference. This lets the
# debug/release phony targets switch optimization/defines via a target-specific BUILD
# without a recursive $(MAKE) submake (which triggered the "forced in submake" jobserver
# warning and a redundant second parallel pass). LDFLAGS is unused (empty for both).
CFLAGS = $(if $(filter debug,$(BUILD)),-std=c++23 -O0 -g -DDEBUG,-std=c++23 -O3 -march=native -DNDEBUG -pthread -DUSE_IMGUI) -pthread -Wall -Wshadow -isystem third_party/imgui
# Auto header dependencies (-MMD -MP): every compile emits a .d file next to
# its .o, and the include below rebuilds dependents when a header changes.
# Without this, editing a header (e.g. adding a class member) silently leaves
# stale objects with mismatched layouts — a proven source of heap corruption
# and phantom validation errors in this codebase.
DEPFLAGS = -MMD -MP
# NOTE: DEPS/-include live after OBJS is defined (see below); placing the
# include earlier expands to empty and silently disables header tracking.

# Use vendored Vulkan 1.4 SDK headers (LunarG SDK include/), then pkg-config for
# GLFW and the Vulkan loader. The vulkan headers are listed first so they shadow
# the system 1.3 headers from libvulkan-dev. Also add common ImGui/stb includes.
VK_SDK_INCLUDE = -Ithird_party/Vulkan-Headers/include
INCLUDES = $(VK_SDK_INCLUDE) `pkg-config --cflags glfw3 vulkan` -I. -isystem third_party/imgui -isystem third_party/imgui/backends -I/usr/include/stb
LIBS = `pkg-config --libs glfw3 vulkan` -lstb -ljpeg -lgdal -lz

# Wii Remote support via vendored wiiuse (third_party/wiiuse).
# Auto-detects BlueZ so the library can be compiled even if not installed.
HAS_WIIUSE := 1
WIIUSE_SRC := third_party/wiiuse/src
WIIUSE_SRCS := $(WIIUSE_SRC)/classic.c $(WIIUSE_SRC)/dynamics.c $(WIIUSE_SRC)/events.c \
                $(WIIUSE_SRC)/guitar_hero_3.c $(WIIUSE_SRC)/io.c $(WIIUSE_SRC)/ir.c \
                $(WIIUSE_SRC)/nunchuk.c $(WIIUSE_SRC)/wiiuse.c $(WIIUSE_SRC)/wiiboard.c \
                $(WIIUSE_SRC)/motion_plus.c $(WIIUSE_SRC)/os_nix.c $(WIIUSE_SRC)/tatacon.c \
                $(WIIUSE_SRC)/util.c
WIIUSE_OBJS = $(patsubst third_party/wiiuse/src/%.c,$(OBJ_DIR)/wiiuse/%.o,$(WIIUSE_SRCS))
INCLUDES += -I$(WIIUSE_SRC)
LIBS += $(shell pkg-config --libs bluez) -lm -lrt
CFLAGS += -DHAS_WIIUSE -DWIIUSE_STATIC -DWIIUSE_COMPILE_LIB

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
SRCS := $(wildcard MyApp.cpp world/*.cpp utils/*.cpp vulkan/*.cpp vulkan/renderer/*.cpp vulkan/streaming/*.cpp widgets/*.cpp widgets/components/*.cpp events/*.cpp math/*.cpp sdf/*.cpp space/*.cpp services/*.cpp) third_party/miniaudio/miniaudio_impl.cpp
# Exclude legacy utils Camera implementation (migrated to math/Camera)
SRCS := $(filter-out utils/Camera.cpp,$(SRCS))
OBJ_DIR := $(OUT_DIR)/obj

# Compose object lists, then forcibly filter out any absolute /imgui/*.o
OBJS := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(SRCS)) $(IMGUI_OBJS) $(WIIUSE_OBJS)

OUT = $(OUT_DIR)/app

# Objects used for the standalone server (exclude the app's MyApp.o and vulkan objects to avoid duplicate main
# and linking against Vulkan)
SERVER_OBJS := $(filter-out $(OBJ_DIR)/MyApp.o $(OBJ_DIR)/vulkan/%.o $(OBJ_DIR)/widgets/%.o $(OBJ_DIR)/services/%.o $(OBJ_DIR)/events/KeyboardPublisher.o $(OBJ_DIR)/events/GamepadPublisher.o $(OBJ_DIR)/events/MousePublisher.o $(OBJ_DIR)/events/RadialMenuHandler.o,$(OBJS))

# Server-specific link flags: now include glfw and vulkan libs for ImGui backends
SERVER_LIBS := $(LIBS)
SERVER_INCLUDES := -isystem third_party/imgui -isystem third_party/imgui/backends -I/usr/include/stb

# Header dependencies generated per-TU by -MMD -MP (see DEPFLAGS above).
# MUST stay after OBJS is defined; otherwise the list expands empty and
# header edits silently leave stale objects (mismatched class layouts).
DEPS := $(OBJS:.o=.d)
-include $(DEPS)


# Automatically find all shader source files in shaders/ with known extensions
# rgen/rmiss/rchit/rint/rahit/rcallable are the KHR ray-tracing stages used by
# the hybrid-RT water pipeline (compiled with the same vulkan1.3 target; glslc
# enables GL_EXT_ray_tracing / GL_EXT_ray_query per-shader via #extension).
SHADER_EXTS = vert frag geom comp tesc tese
RT_SHADER_EXTS = rgen rmiss rchit rint rahit rcallable
SHADERS = $(foreach ext,$(SHADER_EXTS),$(wildcard shaders/*.$(ext)))
SHADER_INCLUDES = $(wildcard shaders/includes/*.glsl)
# Map each shader to its corresponding .spv output in bin/shaders, preserving extension
OUT_SPVS = \
	$(patsubst shaders/%.vert, $(OUT_DIR)/shaders/%.vert.spv, $(wildcard shaders/*.vert)) \
	$(patsubst shaders/%.frag, $(OUT_DIR)/shaders/%.frag.spv, $(wildcard shaders/*.frag)) \
	$(patsubst shaders/%.geom, $(OUT_DIR)/shaders/%.geom.spv, $(wildcard shaders/*.geom)) \
	$(patsubst shaders/%.comp, $(OUT_DIR)/shaders/%.comp.spv, $(wildcard shaders/*.comp)) \
	$(patsubst shaders/%.tesc, $(OUT_DIR)/shaders/%.tesc.spv, $(wildcard shaders/*.tesc)) \
	$(patsubst shaders/%.tese, $(OUT_DIR)/shaders/%.tese.spv, $(wildcard shaders/*.tese)) \
	$(foreach ext,$(RT_SHADER_EXTS),$(patsubst shaders/%.$(ext), $(OUT_DIR)/shaders/%.$(ext).spv, $(wildcard shaders/*.$(ext)))) \
	$(OUT_DIR)/shaders/main_brush.frag.spv \
	$(OUT_DIR)/shaders/main_rt.frag.spv \
	$(OUT_DIR)/shaders/main_water.frag.spv \
	$(OUT_DIR)/shaders/main_water_rt.frag.spv \
	$(OUT_DIR)/shaders/main_water.vert.spv \
	$(OUT_DIR)/shaders/main_water.tesc.spv \
	$(OUT_DIR)/shaders/main_water.tese.spv

# Compile main.frag with -DBRUSH_PASS for brush rendering (no PAINT mode, no set=1)
$(OUT_DIR)/shaders/main_brush.frag.spv: shaders/main.frag $(SHADER_INCLUDES)
	@echo "Compiling shader: $< -> $@ (BRUSH_PASS)"
	@mkdir -p $(dir $@)
	@if command -v glslc >/dev/null 2>&1; then \
		glslc --target-env=vulkan1.3 -Ishaders/includes -DBRUSH_PASS $< -o $@; \
	else \
		glslangValidator -Ishaders/includes -V --target-env vulkan1.3 --D BRUSH_PASS $< -o $@; \
	fi

# Hybrid RT variants (hardware ray tracing: ray queries + TLAS). Selected at
# runtime by rayTracingEnabled(); the non-RT variants above stay the fallback
# for hardware without VK_KHR_ray_query (validation-clean, sky approx).
$(OUT_DIR)/shaders/main_rt.frag.spv: shaders/main.frag $(SHADER_INCLUDES)
	@echo "Compiling shader: $< -> $@ (RT_ENABLED)"
	@mkdir -p $(dir $@)
	@if command -v glslc >/dev/null 2>&1; then \
		glslc --target-env=vulkan1.3 -Ishaders/includes -DRT_ENABLED $< -o $@; \
	else \
		glslangValidator -Ishaders/includes -V --target-env vulkan1.3 --D RT_ENABLED $< -o $@; \
	fi
# Phase-1 merged water stages: same main.* sources compiled with
# -DWATER_MODE=1 (water varyings/bindings/geometry paths). The fragment gets
# an RT and a non-RT variant, mirroring main_rt/main.
$(OUT_DIR)/shaders/main_water.frag.spv: shaders/main.frag $(SHADER_INCLUDES)
	@echo "Compiling shader: $< -> $@ (WATER_MODE=1)"
	@mkdir -p $(dir $@)
	@if command -v glslc >/dev/null 2>&1; then \
		glslc --target-env=vulkan1.3 -Ishaders/includes -DWATER_MODE=1 $< -o $@; \
	else \
		glslangValidator -Ishaders/includes -V --target-env vulkan1.3 --D WATER_MODE=1 $< -o $@; \
	fi
$(OUT_DIR)/shaders/main_water_rt.frag.spv: shaders/main.frag $(SHADER_INCLUDES)
	@echo "Compiling shader: $< -> $@ (WATER_MODE=1 RT_ENABLED)"
	@mkdir -p $(dir $@)
	@if command -v glslc >/dev/null 2>&1; then \
		glslc --target-env=vulkan1.3 -Ishaders/includes -DWATER_MODE=1 -DRT_ENABLED $< -o $@; \
	else \
		glslangValidator -Ishaders/includes -V --target-env vulkan1.3 --D WATER_MODE=1 --D RT_ENABLED $< -o $@; \
	fi
$(OUT_DIR)/shaders/main_water.vert.spv: shaders/main.vert $(SHADER_INCLUDES)
	@echo "Compiling shader: $< -> $@ (WATER_MODE=1)"
	@mkdir -p $(dir $@)
	@if command -v glslc >/dev/null 2>&1; then \
		glslc --target-env=vulkan1.3 -Ishaders/includes -DWATER_MODE=1 $< -o $@; \
	else \
		glslangValidator -Ishaders/includes -V --target-env vulkan1.3 --D WATER_MODE=1 $< -o $@; \
	fi
$(OUT_DIR)/shaders/main_water.tesc.spv: shaders/main.tesc $(SHADER_INCLUDES)
	@echo "Compiling shader: $< -> $@ (WATER_MODE=1)"
	@mkdir -p $(dir $@)
	@if command -v glslc >/dev/null 2>&1; then \
		glslc --target-env=vulkan1.3 -Ishaders/includes -DWATER_MODE=1 $< -o $@; \
	else \
		glslangValidator -Ishaders/includes -V --target-env vulkan1.3 --D WATER_MODE=1 $< -o $@; \
	fi
$(OUT_DIR)/shaders/main_water.tese.spv: shaders/main.tese $(SHADER_INCLUDES)
	@echo "Compiling shader: $< -> $@ (WATER_MODE=1)"
	@mkdir -p $(dir $@)
	@if command -v glslc >/dev/null 2>&1; then \
		glslc --target-env=vulkan1.3 -Ishaders/includes -DWATER_MODE=1 $< -o $@; \
	else \
		glslangValidator -Ishaders/includes -V --target-env vulkan1.3 --D WATER_MODE=1 $< -o $@; \
	fi


# Recursively create all object directories needed for all sources
define make-obj-dirs
	@mkdir -p $(OUT_DIR)
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(OBJ_DIR)/imgui
	@mkdir -p $(OBJ_DIR)/imgui/backends
	@find world utils vulkan widgets events math sdf space services -type d 2>/dev/null | while read dir; do \
		mkdir -p $(OBJ_DIR)/$$dir; \
	done
endef

# Sound-wrapped build: the real work lives in _all and is driven through a
# sub-make so a failed build (or link) can still play the error jingle, while
# a successful one plays the success jingle. BUILD is forwarded explicitly so
# `make debug` / `make release` keep their target-specific configuration.
all:
	@$(call PLAY_SOUND,build)
	@$(MAKE) --no-print-directory _all BUILD=$(BUILD) \
		|| { $(call PLAY_SOUND,error); exit 1; }
	@$(call PLAY_SOUND,success)

.PHONY: _all
_all: imgui shaders $(OUT) server
	$(call make-obj-dirs)
	@mkdir -p $(OBJ_DIR)/imgui

.PHONY: imgui
imgui: $(IMGUI_OBJS)
	@echo "ImGui compilation complete"

.PHONY: server
server: $(SERVER_OBJS)
	@mkdir -p $(OUT_DIR)
	@$(CC) $(CFLAGS) $(SERVER_INCLUDES) server.cpp $(SERVER_OBJS) -o $(OUT_DIR)/server $(SERVER_LIBS) $(LDFLAGS)

$(OUT): $(OBJS)
	@echo "Linking: $(OUT)"
	@$(CC) $(CFLAGS) $(INCLUDES) $(OBJS) -o $(OUT) $(LIBS) $(LDFLAGS)

	@echo "Copying runtime resources to $(OUT_DIR)/"
	@mkdir -p $(OUT_DIR)/shaders
	@if [ -d textures ]; then cp -a textures $(OUT_DIR)/ || true; fi
	@if [ -d fonts ]; then cp -a fonts $(OUT_DIR)/ || true; fi
	@if [ -d scenes ]; then cp -a scenes $(OUT_DIR)/ || true; fi
	@if [ -f imgui.ini ]; then cp imgui.ini $(OUT_DIR)/ || true; fi
	@date "+%Y-%m-%d %H:%M:%S" > $(OUT_DIR)/build_timestamp.txt

# Pattern rule: compile each .cpp into an object under $(OBJ_DIR), preserving subdirs

# Pattern rule for normal sources
$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	@echo "Compiling: $<"
	@$(CC) $(CFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@


$(OBJ_DIR)/imgui/%.o: third_party/imgui/%.cpp
	@mkdir -p $(OBJ_DIR)/imgui
	@echo "Compiling ImGui: $<"
	@$(CC) $(CFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/imgui/backends/%.o: third_party/imgui/backends/%.cpp
	@mkdir -p $(OBJ_DIR)/imgui/backends
	@echo "Compiling ImGui Backend: $<"
	@$(CC) $(CFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

# Compile vendored wiiuse C sources with gcc (no C++ flags).
$(OBJ_DIR)/wiiuse/%.o: third_party/wiiuse/src/%.c
	@mkdir -p $(dir $@)
	@echo "Compiling wiiuse: $<"
	@gcc -O2 -Wall -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-but-set-variable -Ithird_party/wiiuse/src $(shell pkg-config --cflags bluez) -DWIIUSE_STATIC -DWIIUSE_COMPILE_LIB -c $< -o $@


shaders: $(OUT_SPVS)
	@# Copy compiled SPIR-V back to the source shaders/ folder so FileReader can load shaders/*.spv at runtime
	@mkdir -p shaders
	@cp -u $(OUT_DIR)/shaders/*.spv shaders/ 2>/dev/null || true
	@rm shaders/*.spv 2>/dev/null || true


# Generic pattern rule for all shader extensions in $(SHADER_EXTS)
define SHADER_COMPILE_RULE
$(OUT_DIR)/shaders/%.$(1).spv: shaders/%.$(1) $(SHADER_INCLUDES)
	@echo "Compiling shader: $$< -> $$@"
	@mkdir -p $$(dir $$@)
	@if command -v glslc >/dev/null 2>&1; then \
		glslc --target-env=vulkan1.3 -Ishaders/includes $$< -o $$@; \
	else \
		glslangValidator -Ishaders/includes -V --target-env vulkan1.3 $$< -o $$@; \
	fi
endef

$(foreach ext,$(SHADER_EXTS),$(eval $(call SHADER_COMPILE_RULE,$(ext))))
$(foreach ext,$(RT_SHADER_EXTS),$(eval $(call SHADER_COMPILE_RULE,$(ext))))

.PHONY: debug release


release: BUILD = release
release: all

.PHONY: run run-debug valgrind callgrind
# NOTE: run/run-debug DEPEND on the build (all/debug) so stale binaries or
# shaders can never be launched by accident — a bare `make run` refreshes
# everything first (AGENTS.md documents these as "build + run"). When the app
# exits, the success/error jingle reflects its exit status (bash PIPESTATUS
# keeps the app's code through the tee pipeline).
run: 
	@echo "Running app from $(OUT_DIR)/"
	@$(call PLAY_SOUND,run)
	@mkdir -p logs
	@cd $(OUT_DIR) && bash -c './app 2>&1 | tee ../logs/run.log; exit $${PIPESTATUS[0]}' \
		&& $(call PLAY_SOUND,success) \
		|| $(call PLAY_SOUND,error);

run-debug: debug
	@echo "Running debug build from $(OUT_DIR)/"
	@$(call PLAY_SOUND,run)
	@mkdir -p logs
	@cd $(OUT_DIR) && bash -c './app 2>&1 | tee ../logs/run.log; exit $${PIPESTATUS[0]}' \
		&& $(call PLAY_SOUND,success) \
		|| $(call PLAY_SOUND,error);

valgrind: debug
	@echo "Running valgrind with suppressions..."
	@mkdir -p logs
	@cd $(OUT_DIR) && valgrind --suppressions=../valgrind.supp --leak-check=full --show-leak-kinds=all ./app > ../logs/valgrind.log 2>&1; echo "Exit code: $$?"

clean:
	@if [ "$(RESET)" = "1" ]; then reset; fi
	# Remove bin/ directory
	rm -rf $(OUT_DIR)
	# Remove generated SPIR-V files in shaders/ (if present)
	-rm -f $(SPVS)
	rm -f pipeline_cache.bin
	@$(call PLAY_SOUND,clean)

debug: BUILD = debug
debug: all
	
install:
	sudo apt install vulkan-validationlayers
	sudo apt install glslang-tools
	sudo apt-get install robin-map-dev
	
	# 1. Install dependencies
	sudo apt update
	sudo apt install -y build-essential \
						git cmake pkg-config \
						libglfw3-dev \
						libvulkan-dev \
						vulkan-validationlayers \
						glslang-tools \
						libglm-dev \
						libshaderc-dev \
						libstb-dev \
						libgdal-dev \
						libbluetooth-dev \
						vulkan-tools
	# Wiimote support uses the vendored wiiuse library (third_party/wiiuse)
	# 2. Clone Dear ImGui
	mkdir -p third_party

	cd third_party
	git clone https://github.com/ocornut/imgui.git
	git clone https://github.com/mackron/miniaudio.git
	git clone https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git

	# Vulkan 1.4 SDK headers (the build uses third_party/Vulkan-Headers/include).
	# Pinned to the same SDK tag used by the vendored headers so the API version
	# matches VK_API_VERSION_1_4 targeted by the engine. Re-clone if missing or
	# not a real git repo (e.g. a plain vendored checkout).
	if [ ! -d Vulkan-Headers/.git ]; then \
		rm -rf Vulkan-Headers; \
		git clone https://github.com/KhronosGroup/Vulkan-Headers.git; \
	fi
	cd Vulkan-Headers && git checkout vulkan-sdk-1.4.357.0 && cd ..

	@if [ ! -f third_party/miniaudio/miniaudio_impl.cpp ]; then \
		printf '#define MINIAUDIO_IMPLEMENTATION\n#include "miniaudio.h"\n' > third_party/miniaudio/miniaudio_impl.cpp; \
	fi
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

	# 6. Install Vulkan 1.4 SDK headers system-wide. The project targets Vulkan 1.4
	# but the distro libvulkan-dev ships 1.3 headers, so we install the vendored
	# 1.4 headers (third_party/Vulkan-Headers) into /usr/local/include/vulkan.
	# /usr/local/include is searched before /usr/include, so this shadows the
	# older 1.3 headers for any build that resolves Vulkan via the system path.
	sudo mkdir -p /usr/local/include/vulkan
	sudo cp -r third_party/Vulkan-Headers/include/vulkan/* /usr/local/include/vulkan/
	sudo ldconfig

	@echo "Optionally, for profiling and code analysis tools, run:"
	@echo "  sudo apt-get install cloc kcachegrind massif-visualizer"

.PHONY: cloc
cloc:
	@echo "Running cloc to count lines of code..."
	@# Exclude runtime bins and third_party from the count; print to terminal (no file)
	@if command -v cloc >/dev/null 2>&1; then \
		cloc --exclude-dir=$(OUT_DIR),third_party --exclude-ext=tex .; \
	else \
		echo "cloc not found on PATH. Install it (e.g. sudo apt install cloc) to get a detailed LOC report."; \
	fi

callgrind: debug
	@echo "Running valgrind callgrind profiler on $(OUT_DIR)/app..."
	@mkdir -p logs
	@cd $(OUT_DIR) && valgrind --tool=callgrind --callgrind-out-file=../logs/callgrind.out ./app 2>&1 | tee ../logs/callgrind.log; echo "Exit code: $${PIPESTATUS[0]}"
	@echo "Profile written to logs/callgrind.out"
	@if command -v kcachegrind >/dev/null 2>&1; then \
		kcachegrind logs/callgrind.out; \
	else \
		echo "kcachegrind not found on PATH. Install it (e.g. sudo apt install kcachegrind) to visualize logs/callgrind.out."; \
	fi
