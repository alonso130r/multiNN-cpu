# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/aarch64/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/aarch64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/vijaygoyal/Documents/GitHub/gpt-cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/vijaygoyal/Documents/GitHub/gpt-cpp/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/gpt_cpp.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/gpt_cpp.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/gpt_cpp.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/gpt_cpp.dir/flags.make

CMakeFiles/gpt_cpp.dir/singleNN.cpp.o: CMakeFiles/gpt_cpp.dir/flags.make
CMakeFiles/gpt_cpp.dir/singleNN.cpp.o: /Users/vijaygoyal/Documents/GitHub/gpt-cpp/singleNN.cpp
CMakeFiles/gpt_cpp.dir/singleNN.cpp.o: CMakeFiles/gpt_cpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/vijaygoyal/Documents/GitHub/gpt-cpp/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/gpt_cpp.dir/singleNN.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/gpt_cpp.dir/singleNN.cpp.o -MF CMakeFiles/gpt_cpp.dir/singleNN.cpp.o.d -o CMakeFiles/gpt_cpp.dir/singleNN.cpp.o -c /Users/vijaygoyal/Documents/GitHub/gpt-cpp/singleNN.cpp

CMakeFiles/gpt_cpp.dir/singleNN.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/gpt_cpp.dir/singleNN.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/vijaygoyal/Documents/GitHub/gpt-cpp/singleNN.cpp > CMakeFiles/gpt_cpp.dir/singleNN.cpp.i

CMakeFiles/gpt_cpp.dir/singleNN.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/gpt_cpp.dir/singleNN.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/vijaygoyal/Documents/GitHub/gpt-cpp/singleNN.cpp -o CMakeFiles/gpt_cpp.dir/singleNN.cpp.s

CMakeFiles/gpt_cpp.dir/multi-class-NN/neuron.cpp.o: CMakeFiles/gpt_cpp.dir/flags.make
CMakeFiles/gpt_cpp.dir/multi-class-NN/neuron.cpp.o: /Users/vijaygoyal/Documents/GitHub/gpt-cpp/multi-class-NN/neuron.cpp
CMakeFiles/gpt_cpp.dir/multi-class-NN/neuron.cpp.o: CMakeFiles/gpt_cpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/vijaygoyal/Documents/GitHub/gpt-cpp/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/gpt_cpp.dir/multi-class-NN/neuron.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/gpt_cpp.dir/multi-class-NN/neuron.cpp.o -MF CMakeFiles/gpt_cpp.dir/multi-class-NN/neuron.cpp.o.d -o CMakeFiles/gpt_cpp.dir/multi-class-NN/neuron.cpp.o -c /Users/vijaygoyal/Documents/GitHub/gpt-cpp/multi-class-NN/neuron.cpp

CMakeFiles/gpt_cpp.dir/multi-class-NN/neuron.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/gpt_cpp.dir/multi-class-NN/neuron.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/vijaygoyal/Documents/GitHub/gpt-cpp/multi-class-NN/neuron.cpp > CMakeFiles/gpt_cpp.dir/multi-class-NN/neuron.cpp.i

CMakeFiles/gpt_cpp.dir/multi-class-NN/neuron.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/gpt_cpp.dir/multi-class-NN/neuron.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/vijaygoyal/Documents/GitHub/gpt-cpp/multi-class-NN/neuron.cpp -o CMakeFiles/gpt_cpp.dir/multi-class-NN/neuron.cpp.s

CMakeFiles/gpt_cpp.dir/multi-class-NN/layer.cpp.o: CMakeFiles/gpt_cpp.dir/flags.make
CMakeFiles/gpt_cpp.dir/multi-class-NN/layer.cpp.o: /Users/vijaygoyal/Documents/GitHub/gpt-cpp/multi-class-NN/layer.cpp
CMakeFiles/gpt_cpp.dir/multi-class-NN/layer.cpp.o: CMakeFiles/gpt_cpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/vijaygoyal/Documents/GitHub/gpt-cpp/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/gpt_cpp.dir/multi-class-NN/layer.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/gpt_cpp.dir/multi-class-NN/layer.cpp.o -MF CMakeFiles/gpt_cpp.dir/multi-class-NN/layer.cpp.o.d -o CMakeFiles/gpt_cpp.dir/multi-class-NN/layer.cpp.o -c /Users/vijaygoyal/Documents/GitHub/gpt-cpp/multi-class-NN/layer.cpp

CMakeFiles/gpt_cpp.dir/multi-class-NN/layer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/gpt_cpp.dir/multi-class-NN/layer.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/vijaygoyal/Documents/GitHub/gpt-cpp/multi-class-NN/layer.cpp > CMakeFiles/gpt_cpp.dir/multi-class-NN/layer.cpp.i

CMakeFiles/gpt_cpp.dir/multi-class-NN/layer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/gpt_cpp.dir/multi-class-NN/layer.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/vijaygoyal/Documents/GitHub/gpt-cpp/multi-class-NN/layer.cpp -o CMakeFiles/gpt_cpp.dir/multi-class-NN/layer.cpp.s

CMakeFiles/gpt_cpp.dir/multi-class-NN/layerWrappers.cpp.o: CMakeFiles/gpt_cpp.dir/flags.make
CMakeFiles/gpt_cpp.dir/multi-class-NN/layerWrappers.cpp.o: /Users/vijaygoyal/Documents/GitHub/gpt-cpp/multi-class-NN/layerWrappers.cpp
CMakeFiles/gpt_cpp.dir/multi-class-NN/layerWrappers.cpp.o: CMakeFiles/gpt_cpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/vijaygoyal/Documents/GitHub/gpt-cpp/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/gpt_cpp.dir/multi-class-NN/layerWrappers.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/gpt_cpp.dir/multi-class-NN/layerWrappers.cpp.o -MF CMakeFiles/gpt_cpp.dir/multi-class-NN/layerWrappers.cpp.o.d -o CMakeFiles/gpt_cpp.dir/multi-class-NN/layerWrappers.cpp.o -c /Users/vijaygoyal/Documents/GitHub/gpt-cpp/multi-class-NN/layerWrappers.cpp

CMakeFiles/gpt_cpp.dir/multi-class-NN/layerWrappers.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/gpt_cpp.dir/multi-class-NN/layerWrappers.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/vijaygoyal/Documents/GitHub/gpt-cpp/multi-class-NN/layerWrappers.cpp > CMakeFiles/gpt_cpp.dir/multi-class-NN/layerWrappers.cpp.i

CMakeFiles/gpt_cpp.dir/multi-class-NN/layerWrappers.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/gpt_cpp.dir/multi-class-NN/layerWrappers.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/vijaygoyal/Documents/GitHub/gpt-cpp/multi-class-NN/layerWrappers.cpp -o CMakeFiles/gpt_cpp.dir/multi-class-NN/layerWrappers.cpp.s

CMakeFiles/gpt_cpp.dir/multi-class-NN/neuralNetwork.cpp.o: CMakeFiles/gpt_cpp.dir/flags.make
CMakeFiles/gpt_cpp.dir/multi-class-NN/neuralNetwork.cpp.o: /Users/vijaygoyal/Documents/GitHub/gpt-cpp/multi-class-NN/neuralNetwork.cpp
CMakeFiles/gpt_cpp.dir/multi-class-NN/neuralNetwork.cpp.o: CMakeFiles/gpt_cpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/vijaygoyal/Documents/GitHub/gpt-cpp/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/gpt_cpp.dir/multi-class-NN/neuralNetwork.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/gpt_cpp.dir/multi-class-NN/neuralNetwork.cpp.o -MF CMakeFiles/gpt_cpp.dir/multi-class-NN/neuralNetwork.cpp.o.d -o CMakeFiles/gpt_cpp.dir/multi-class-NN/neuralNetwork.cpp.o -c /Users/vijaygoyal/Documents/GitHub/gpt-cpp/multi-class-NN/neuralNetwork.cpp

CMakeFiles/gpt_cpp.dir/multi-class-NN/neuralNetwork.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/gpt_cpp.dir/multi-class-NN/neuralNetwork.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/vijaygoyal/Documents/GitHub/gpt-cpp/multi-class-NN/neuralNetwork.cpp > CMakeFiles/gpt_cpp.dir/multi-class-NN/neuralNetwork.cpp.i

CMakeFiles/gpt_cpp.dir/multi-class-NN/neuralNetwork.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/gpt_cpp.dir/multi-class-NN/neuralNetwork.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/vijaygoyal/Documents/GitHub/gpt-cpp/multi-class-NN/neuralNetwork.cpp -o CMakeFiles/gpt_cpp.dir/multi-class-NN/neuralNetwork.cpp.s

CMakeFiles/gpt_cpp.dir/multi-class-NN/trainTest1.cpp.o: CMakeFiles/gpt_cpp.dir/flags.make
CMakeFiles/gpt_cpp.dir/multi-class-NN/trainTest1.cpp.o: /Users/vijaygoyal/Documents/GitHub/gpt-cpp/multi-class-NN/trainTest1.cpp
CMakeFiles/gpt_cpp.dir/multi-class-NN/trainTest1.cpp.o: CMakeFiles/gpt_cpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/vijaygoyal/Documents/GitHub/gpt-cpp/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/gpt_cpp.dir/multi-class-NN/trainTest1.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/gpt_cpp.dir/multi-class-NN/trainTest1.cpp.o -MF CMakeFiles/gpt_cpp.dir/multi-class-NN/trainTest1.cpp.o.d -o CMakeFiles/gpt_cpp.dir/multi-class-NN/trainTest1.cpp.o -c /Users/vijaygoyal/Documents/GitHub/gpt-cpp/multi-class-NN/trainTest1.cpp

CMakeFiles/gpt_cpp.dir/multi-class-NN/trainTest1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/gpt_cpp.dir/multi-class-NN/trainTest1.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/vijaygoyal/Documents/GitHub/gpt-cpp/multi-class-NN/trainTest1.cpp > CMakeFiles/gpt_cpp.dir/multi-class-NN/trainTest1.cpp.i

CMakeFiles/gpt_cpp.dir/multi-class-NN/trainTest1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/gpt_cpp.dir/multi-class-NN/trainTest1.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/vijaygoyal/Documents/GitHub/gpt-cpp/multi-class-NN/trainTest1.cpp -o CMakeFiles/gpt_cpp.dir/multi-class-NN/trainTest1.cpp.s

# Object files for target gpt_cpp
gpt_cpp_OBJECTS = \
"CMakeFiles/gpt_cpp.dir/singleNN.cpp.o" \
"CMakeFiles/gpt_cpp.dir/multi-class-NN/neuron.cpp.o" \
"CMakeFiles/gpt_cpp.dir/multi-class-NN/layer.cpp.o" \
"CMakeFiles/gpt_cpp.dir/multi-class-NN/layerWrappers.cpp.o" \
"CMakeFiles/gpt_cpp.dir/multi-class-NN/neuralNetwork.cpp.o" \
"CMakeFiles/gpt_cpp.dir/multi-class-NN/trainTest1.cpp.o"

# External object files for target gpt_cpp
gpt_cpp_EXTERNAL_OBJECTS =

gpt_cpp: CMakeFiles/gpt_cpp.dir/singleNN.cpp.o
gpt_cpp: CMakeFiles/gpt_cpp.dir/multi-class-NN/neuron.cpp.o
gpt_cpp: CMakeFiles/gpt_cpp.dir/multi-class-NN/layer.cpp.o
gpt_cpp: CMakeFiles/gpt_cpp.dir/multi-class-NN/layerWrappers.cpp.o
gpt_cpp: CMakeFiles/gpt_cpp.dir/multi-class-NN/neuralNetwork.cpp.o
gpt_cpp: CMakeFiles/gpt_cpp.dir/multi-class-NN/trainTest1.cpp.o
gpt_cpp: CMakeFiles/gpt_cpp.dir/build.make
gpt_cpp: CMakeFiles/gpt_cpp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/vijaygoyal/Documents/GitHub/gpt-cpp/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable gpt_cpp"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gpt_cpp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/gpt_cpp.dir/build: gpt_cpp
.PHONY : CMakeFiles/gpt_cpp.dir/build

CMakeFiles/gpt_cpp.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/gpt_cpp.dir/cmake_clean.cmake
.PHONY : CMakeFiles/gpt_cpp.dir/clean

CMakeFiles/gpt_cpp.dir/depend:
	cd /Users/vijaygoyal/Documents/GitHub/gpt-cpp/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/vijaygoyal/Documents/GitHub/gpt-cpp /Users/vijaygoyal/Documents/GitHub/gpt-cpp /Users/vijaygoyal/Documents/GitHub/gpt-cpp/cmake-build-debug /Users/vijaygoyal/Documents/GitHub/gpt-cpp/cmake-build-debug /Users/vijaygoyal/Documents/GitHub/gpt-cpp/cmake-build-debug/CMakeFiles/gpt_cpp.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/gpt_cpp.dir/depend

