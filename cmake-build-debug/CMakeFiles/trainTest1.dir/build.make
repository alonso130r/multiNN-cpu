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
include CMakeFiles/trainTest1.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/trainTest1.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/trainTest1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/trainTest1.dir/flags.make

CMakeFiles/trainTest1.dir/multi-class-NN/trainTest1.cpp.o: CMakeFiles/trainTest1.dir/flags.make
CMakeFiles/trainTest1.dir/multi-class-NN/trainTest1.cpp.o: /Users/vijaygoyal/Documents/GitHub/gpt-cpp/multi-class-NN/trainTest1.cpp
CMakeFiles/trainTest1.dir/multi-class-NN/trainTest1.cpp.o: CMakeFiles/trainTest1.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/vijaygoyal/Documents/GitHub/gpt-cpp/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/trainTest1.dir/multi-class-NN/trainTest1.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/trainTest1.dir/multi-class-NN/trainTest1.cpp.o -MF CMakeFiles/trainTest1.dir/multi-class-NN/trainTest1.cpp.o.d -o CMakeFiles/trainTest1.dir/multi-class-NN/trainTest1.cpp.o -c /Users/vijaygoyal/Documents/GitHub/gpt-cpp/multi-class-NN/trainTest1.cpp

CMakeFiles/trainTest1.dir/multi-class-NN/trainTest1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/trainTest1.dir/multi-class-NN/trainTest1.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/vijaygoyal/Documents/GitHub/gpt-cpp/multi-class-NN/trainTest1.cpp > CMakeFiles/trainTest1.dir/multi-class-NN/trainTest1.cpp.i

CMakeFiles/trainTest1.dir/multi-class-NN/trainTest1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/trainTest1.dir/multi-class-NN/trainTest1.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/vijaygoyal/Documents/GitHub/gpt-cpp/multi-class-NN/trainTest1.cpp -o CMakeFiles/trainTest1.dir/multi-class-NN/trainTest1.cpp.s

# Object files for target trainTest1
trainTest1_OBJECTS = \
"CMakeFiles/trainTest1.dir/multi-class-NN/trainTest1.cpp.o"

# External object files for target trainTest1
trainTest1_EXTERNAL_OBJECTS =

trainTest1: CMakeFiles/trainTest1.dir/multi-class-NN/trainTest1.cpp.o
trainTest1: CMakeFiles/trainTest1.dir/build.make
trainTest1: CMakeFiles/trainTest1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/vijaygoyal/Documents/GitHub/gpt-cpp/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable trainTest1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/trainTest1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/trainTest1.dir/build: trainTest1
.PHONY : CMakeFiles/trainTest1.dir/build

CMakeFiles/trainTest1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/trainTest1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/trainTest1.dir/clean

CMakeFiles/trainTest1.dir/depend:
	cd /Users/vijaygoyal/Documents/GitHub/gpt-cpp/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/vijaygoyal/Documents/GitHub/gpt-cpp /Users/vijaygoyal/Documents/GitHub/gpt-cpp /Users/vijaygoyal/Documents/GitHub/gpt-cpp/cmake-build-debug /Users/vijaygoyal/Documents/GitHub/gpt-cpp/cmake-build-debug /Users/vijaygoyal/Documents/GitHub/gpt-cpp/cmake-build-debug/CMakeFiles/trainTest1.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/trainTest1.dir/depend

