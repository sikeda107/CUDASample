# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/m5211129/GitRepository/CUDASample/CMAKE_CUDA

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/m5211129/GitRepository/CUDASample/CMAKE_CUDA/cmake-build

# Include any dependencies generated for this target.
include CMakeFiles/Main.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Main.dir/flags.make

CMakeFiles/Main.dir/src/main.cpp.o: CMakeFiles/Main.dir/flags.make
CMakeFiles/Main.dir/src/main.cpp.o: ../src/main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/m5211129/GitRepository/CUDASample/CMAKE_CUDA/cmake-build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/Main.dir/src/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/Main.dir/src/main.cpp.o -c /home/m5211129/GitRepository/CUDASample/CMAKE_CUDA/src/main.cpp

CMakeFiles/Main.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Main.dir/src/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/m5211129/GitRepository/CUDASample/CMAKE_CUDA/src/main.cpp > CMakeFiles/Main.dir/src/main.cpp.i

CMakeFiles/Main.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Main.dir/src/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/m5211129/GitRepository/CUDASample/CMAKE_CUDA/src/main.cpp -o CMakeFiles/Main.dir/src/main.cpp.s

CMakeFiles/Main.dir/src/main.cpp.o.requires:
.PHONY : CMakeFiles/Main.dir/src/main.cpp.o.requires

CMakeFiles/Main.dir/src/main.cpp.o.provides: CMakeFiles/Main.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/Main.dir/build.make CMakeFiles/Main.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/Main.dir/src/main.cpp.o.provides

CMakeFiles/Main.dir/src/main.cpp.o.provides.build: CMakeFiles/Main.dir/src/main.cpp.o

# Object files for target Main
Main_OBJECTS = \
"CMakeFiles/Main.dir/src/main.cpp.o"

# External object files for target Main
Main_EXTERNAL_OBJECTS =

Main: CMakeFiles/Main.dir/src/main.cpp.o
Main: CMakeFiles/Main.dir/build.make
Main: CudaVecDouble/libCudaVecDouble.a
Main: /usr/local/cuda-10.0/lib64/libcudart.so
Main: CMakeFiles/Main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable Main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Main.dir/build: Main
.PHONY : CMakeFiles/Main.dir/build

CMakeFiles/Main.dir/requires: CMakeFiles/Main.dir/src/main.cpp.o.requires
.PHONY : CMakeFiles/Main.dir/requires

CMakeFiles/Main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Main.dir/clean

CMakeFiles/Main.dir/depend:
	cd /home/m5211129/GitRepository/CUDASample/CMAKE_CUDA/cmake-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/m5211129/GitRepository/CUDASample/CMAKE_CUDA /home/m5211129/GitRepository/CUDASample/CMAKE_CUDA /home/m5211129/GitRepository/CUDASample/CMAKE_CUDA/cmake-build /home/m5211129/GitRepository/CUDASample/CMAKE_CUDA/cmake-build /home/m5211129/GitRepository/CUDASample/CMAKE_CUDA/cmake-build/CMakeFiles/Main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Main.dir/depend

