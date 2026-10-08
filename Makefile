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
LIBS = `pkg-config --libs glfw3 vulkan` -lstb -ljpeg -lz

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
SRCS := $(wildcard MyApp.cpp world/*.cpp utils/*.cpp vulkan/*.cpp vulkan/core/*.cpp vulkan/resources/*.cpp vulkan/pipeline/*.cpp vulkan/renderer/*.cpp vulkan/renderer/*/*.cpp vulkan/streaming/*.cpp widgets/*.cpp widgets/components/*.cpp events/*.cpp math/*.cpp sdf/*.cpp sdf/gpu/*.cpp sdf/types/*.cpp space/*.cpp services/*.cpp) third_party/miniaudio/miniaudio_impl.cpp
OBJ_DIR := $(OUT_DIR)/obj

# Compose object lists, then forcibly filter out any absolute /imgui/*.o
OBJS := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(SRCS)) $(IMGUI_OBJS) $(WIIUSE_OBJS)

OUT = $(OUT_DIR)/app

# Objects used for the standalone server (exclude the app's MyApp.o and vulkan objects to avoid duplicate main
# and linking against Vulkan)
SERVER_OBJS := $(filter-out $(OBJ_DIR)/MyApp.o $(OBJ_DIR)/vulkan/%.o $(OBJ_DIR)/widgets/%.o $(OBJ_DIR)/services/%.o $(OBJ_DIR)/events/KeyboardPublisher.o $(OBJ_DIR)/events/GamepadPublisher.o $(OBJ_DIR)/events/MousePublisher.o $(OBJ_DIR)/events/RadialMenuHandler.o,$(OBJS))

# Server-specific link flags: now include glfw and vulkan libs for ImGui backends
SERVER_LIBS := $(LIBS)
SERVER_INCLUDES := -isystem third_party/imgui -isystem third_party/imgui/backends -I/usr/include/stb -I.

# Header dependencies generated per-TU by -MMD -MP (see DEPFLAGS above).
# MUST stay after OBJS is defined; otherwise the list expands empty and
# header edits silently leave stale objects (mismatched class layouts).
DEPS := $(OBJS:.o=.d)
-include $(DEPS)


# Automatically find all shader source files under shaders/ (any depth) with
# known extensions. rgen/rmiss/rchit/rint/rahit/rcallable are the KHR
# ray-tracing stages used by the hybrid-RT water pipeline (compiled with the
# same vulkan1.3 target; glslc enables GL_EXT_ray_tracing / GL_EXT_ray_query
# per-shader via #extension). The layout mirrors vulkan/: per-renderer sources
# live under shaders/renderer/<subsystem>/.
SHADER_EXTS = vert frag geom comp tesc tese
RT_SHADER_EXTS = rgen rmiss rchit rint rahit rcallable
SHADER_FILES := $(shell find shaders -type f \( \
	-name '*.vert' -o -name '*.frag' -o -name '*.geom' -o -name '*.comp' \
	-o -name '*.tesc' -o -name '*.tese' -o -name '*.rgen' -o -name '*.rmiss' \
	-o -name '*.rchit' -o -name '*.rint' -o -name '*.rahit' -o -name '*.rcallable' \))
SHADER_INCLUDES = $(wildcard shaders/includes/*.glsl shaders/includes/*/*.glsl shaders/ubo/*.glsl shaders/ssbo/*.glsl shaders/types/*.glsl)
# glslc optimization flag for the WATER shader modules: -O runs the SPIR-V
# optimizer (DCE, constant folding, register-friendly codegen) on the modules
# whose per-pixel cost matters most. It is deliberately NOT applied to the
# generic rule (solid/vegetation/debug shaders) or to the auxiliary water
# fragment shaders compiled by it: spirv-opt prunes statically-unused fragment
# inputs, which turns the extra producer outputs into VVL "declared to output
# location N but is not an Input" warnings. glslangValidator has no equivalent
# flag, so its fallback command lines stay unoptimized.
GLSL_OPT = -O
# Map each shader to its .spv output in bin/shaders, preserving the
# renderer/<subsystem>/ path so bin/shaders mirrors the source tree. Shader
# base names match the owning C++ class (PascalCase); the explicit entries are
# compile-time variants (a per-renderer define) that discovery cannot find.
OUT_SPVS = \
	$(patsubst shaders/%,$(OUT_DIR)/shaders/%.spv,$(SHADER_FILES)) \
	$(OUT_DIR)/shaders/renderer/solid/SolidRendererNoTess.vert.spv \
	$(OUT_DIR)/shaders/renderer/solid/SolidRendererRT.frag.spv \
	$(OUT_DIR)/shaders/renderer/solid/SolidRendererRTProf.frag.spv \
	$(OUT_DIR)/shaders/renderer/shadow/ShadowRenderer.tese.spv \
	$(OUT_DIR)/shaders/renderer/shadow/ShadowRendererBlur5.frag.spv \
	$(OUT_DIR)/shaders/renderer/vegetation/ImpostorCapture.vert.spv \
	$(OUT_DIR)/shaders/renderer/water/WaterRendererNoTess.vert.spv \
	$(OUT_DIR)/shaders/renderer/water/WaterRendererRT.frag.spv \
	$(OUT_DIR)/shaders/renderer/water/WaterRendererNoBody.frag.spv \
	$(OUT_DIR)/shaders/renderer/water/WaterRendererRTNoBody.frag.spv \
	$(OUT_DIR)/shaders/renderer/water/WaterRendererRT.tese.spv \
	$(OUT_DIR)/shaders/renderer/water/WaterRendererRTProf.frag.spv \
	$(OUT_DIR)/shaders/renderer/water/WaterRendererRTProf.tese.spv

# ── Compile-time shader variants ──────────────────────────────────────────
# Each variant reuses a base source plus a define that selects an alternate
# stage interface or feature set. Shader base names match the owning renderer
# class; in-class pass qualifiers are concatenated PascalCase:
#   SolidRenderer*      -> SolidRenderer (also bound by Shadow)
#   WaterRenderer*      -> WaterRenderer (also bound by WaterBackFaceRenderer)
#   ShadowRenderer*     -> ShadowRenderer
#   VegetationRenderer* -> VegetationRenderer
#   shared/no-owner     -> Fullscreen.vert, DepthOnly.frag
# compile_shader <source> <output> <glslc defines> <glslang defines> <opt>
#   <glslc defines>   e.g. "-DRT_ENABLED -DWATER_NO_BODY=1"    (empty if none)
#   <glslang defines> e.g. "--D RT_ENABLED --D WATER_NO_BODY=1" (empty if none)
#   <opt>             $(GLSL_OPT) for optimized modules, empty otherwise
define compile_shader
	@echo "Compiling shader: $(1) -> $(2)"
	@mkdir -p $(dir $(2))
	@if command -v glslc >/dev/null 2>&1; then \
		glslc --target-env=vulkan1.3 -Ishaders/includes $(5) $(3) $(1) -o $(2); \
	else \
		glslangValidator -Ishaders/includes -V --target-env vulkan1.3 $(4) $(1) -o $(2); \
	fi
endef

# WaterRenderer base stages go through the SPIR-V optimizer because their
# per-pixel cost matters most. Explicit rules override the generic
# WaterRenderer.{vert,frag,tesc,tese} pattern rules below.
$(OUT_DIR)/shaders/renderer/water/WaterRenderer.vert.spv: shaders/renderer/water/WaterRenderer.vert $(SHADER_INCLUDES)
	$(call compile_shader,shaders/renderer/water/WaterRenderer.vert,$@,,,$(GLSL_OPT))
$(OUT_DIR)/shaders/renderer/water/WaterRenderer.frag.spv: shaders/renderer/water/WaterRenderer.frag $(SHADER_INCLUDES)
	$(call compile_shader,shaders/renderer/water/WaterRenderer.frag,$@,,,$(GLSL_OPT))
$(OUT_DIR)/shaders/renderer/water/WaterRenderer.tesc.spv: shaders/renderer/water/WaterRenderer.tesc $(SHADER_INCLUDES)
	$(call compile_shader,shaders/renderer/water/WaterRenderer.tesc,$@,,,$(GLSL_OPT))
$(OUT_DIR)/shaders/renderer/water/WaterRenderer.tese.spv: shaders/renderer/water/WaterRenderer.tese $(SHADER_INCLUDES)
	$(call compile_shader,shaders/renderer/water/WaterRenderer.tese,$@,,,$(GLSL_OPT))

# C1 (perf report 19/21): non-tessellation geometry paths (TRIANGLE_LIST, no
# TCS/TES). They write the fragment interface directly; selected at runtime
# when settings.tessellationEnabled is false.
$(OUT_DIR)/shaders/renderer/solid/SolidRendererNoTess.vert.spv: shaders/renderer/solid/SolidRenderer.vert $(SHADER_INCLUDES)
	$(call compile_shader,shaders/renderer/solid/SolidRenderer.vert,$@,-DSOLID_NO_TESS=1,--D SOLID_NO_TESS=1,$(GLSL_OPT))
$(OUT_DIR)/shaders/renderer/water/WaterRendererNoTess.vert.spv: shaders/renderer/water/WaterRenderer.vert $(SHADER_INCLUDES)
	$(call compile_shader,shaders/renderer/water/WaterRenderer.vert,$@,-DWATER_NO_TESS=1,--D WATER_NO_TESS=1,$(GLSL_OPT))

# SolidRenderer fragment variants: the hybrid RT variants (ray queries +
# TLAS). Non-RT hardware uses plain SolidRenderer.frag.
$(OUT_DIR)/shaders/renderer/solid/SolidRendererRT.frag.spv: shaders/renderer/solid/SolidRenderer.frag $(SHADER_INCLUDES)
	$(call compile_shader,shaders/renderer/solid/SolidRenderer.frag,$@,-DRT_ENABLED,--D RT_ENABLED,)
$(OUT_DIR)/shaders/renderer/solid/SolidRendererRTProf.frag.spv: shaders/renderer/solid/SolidRenderer.frag $(SHADER_INCLUDES)
	$(call compile_shader,shaders/renderer/solid/SolidRenderer.frag,$@,-DRT_ENABLED -DRT_PROFILE,--D RT_ENABLED --D RT_PROFILE,)

# WaterRenderer fragment variants: RT (hardware ray tracing) and WATER_NO_BODY
# (single color attachment, no blur aux outputs). Per-op RT profiling needs the
# shader clock capability and is only created at runtime when supported.
$(OUT_DIR)/shaders/renderer/water/WaterRendererRT.frag.spv: shaders/renderer/water/WaterRenderer.frag $(SHADER_INCLUDES)
	$(call compile_shader,shaders/renderer/water/WaterRenderer.frag,$@,-DRT_ENABLED,--D RT_ENABLED,$(GLSL_OPT))
$(OUT_DIR)/shaders/renderer/water/WaterRendererNoBody.frag.spv: shaders/renderer/water/WaterRenderer.frag $(SHADER_INCLUDES)
	$(call compile_shader,shaders/renderer/water/WaterRenderer.frag,$@,-DWATER_NO_BODY=1,--D WATER_NO_BODY=1,$(GLSL_OPT))
$(OUT_DIR)/shaders/renderer/water/WaterRendererRTNoBody.frag.spv: shaders/renderer/water/WaterRenderer.frag $(SHADER_INCLUDES)
	$(call compile_shader,shaders/renderer/water/WaterRenderer.frag,$@,-DRT_ENABLED -DWATER_NO_BODY=1,--D RT_ENABLED --D WATER_NO_BODY=1,$(GLSL_OPT))
$(OUT_DIR)/shaders/renderer/water/WaterRendererRTProf.frag.spv: shaders/renderer/water/WaterRenderer.frag $(SHADER_INCLUDES)
	$(call compile_shader,shaders/renderer/water/WaterRenderer.frag,$@,-DRT_ENABLED -DRT_PROFILE,--D RT_ENABLED --D RT_PROFILE,$(GLSL_OPT))

# RT water TES: adds the optional inline ray-query water-region depth
# (rt.waterDepth). The prof variant also instruments the TES ray-query site.
$(OUT_DIR)/shaders/renderer/water/WaterRendererRT.tese.spv: shaders/renderer/water/WaterRenderer.tese $(SHADER_INCLUDES)
	$(call compile_shader,shaders/renderer/water/WaterRenderer.tese,$@,-DRT_ENABLED,--D RT_ENABLED,$(GLSL_OPT))
$(OUT_DIR)/shaders/renderer/water/WaterRendererRTProf.tese.spv: shaders/renderer/water/WaterRenderer.tese $(SHADER_INCLUDES)
	$(call compile_shader,shaders/renderer/water/WaterRenderer.tese,$@,-DRT_ENABLED -DRT_PROFILE -DRT_PROFILE_TES,--D RT_ENABLED --D RT_PROFILE --D RT_PROFILE_TES,$(GLSL_OPT))

# H4 (perf report 21): shadow-only solid TES (position-varying outputs only,
# displacement preserved) bound by the tessellated shadow pipeline; and the
# narrower 5-tap EVSM blur for the outer cascades.
$(OUT_DIR)/shaders/renderer/shadow/ShadowRenderer.tese.spv: shaders/renderer/solid/SolidRenderer.tese $(SHADER_INCLUDES)
	$(call compile_shader,shaders/renderer/solid/SolidRenderer.tese,$@,-DSHADOW_PASS=1,--D SHADOW_PASS=1,$(GLSL_OPT))
$(OUT_DIR)/shaders/renderer/shadow/ShadowRendererBlur5.frag.spv: shaders/renderer/shadow/ShadowRendererBlur.frag $(SHADER_INCLUDES)
	$(call compile_shader,shaders/renderer/shadow/ShadowRendererBlur.frag,$@,-DBLUR5=1,--D BLUR5=1,$(GLSL_OPT))

# H4/C2: ImpostorCapture variant of VegetationRenderer.vert (VEG_CAPTURE=1):
# evaluates the height scale locally exactly as before the bake (canonical
# single instance, no aux buffer bound). Built WITHOUT -O like its generic
# sibling so capture output is maximally unchanged.
$(OUT_DIR)/shaders/renderer/vegetation/ImpostorCapture.vert.spv: shaders/renderer/vegetation/VegetationRenderer.vert $(SHADER_INCLUDES)
	$(call compile_shader,shaders/renderer/vegetation/VegetationRenderer.vert,$@,-DVEG_CAPTURE=1,--D VEG_CAPTURE=1,)


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
		cloc --exclude-dir=$(OUT_DIR),third_party,docs --exclude-ext=tex .; \
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
