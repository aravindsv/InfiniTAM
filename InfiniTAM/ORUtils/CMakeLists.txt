#############################
# Specify the project files #
#############################

SET(ORUTILS_HEADERS
Vector.h
Matrix.h
Cholesky.h
MathUtils.h
Image.h
CUDADefines.h
LexicalCast.h
MemoryBlock.h
MemoryBlockPersister.h
PlatformIndependence.h
)

SET(ORUTILS_SOURCES
CUDADefines.cpp
)

# If enabled, enables sensible stack traces on Linux, complete with corresponding source
# code, where available. CUDA errors also produce complete stack traces when this is on.
# If disabled, the error messages degrade gracefully to file/line information.
OPTION(WITH_BACKWARDS_CPP "Build with backwards-cpp stack trace dumping library? (Linux-only)" TRUE)
message(STATUS "backwards-cpp-enhanced stack traces? " ${WITH_BACKWARDS_CPP})


if(WITH_BACKWARDS_CPP)
  # Support 'backward-cpp,' a lean stacktrace printing library for Linux.
  add_definitions(-DWITH_BACKWARDS_CPP)
  add_subdirectory(third_party/backward-cpp)
endif()


#############################
# Specify the source groups #
#############################

SOURCE_GROUP("" FILES ${ORUTILS_HEADERS} ${ORUTILS_SOURCES})
add_library(ORUtils ${ORUTILS_HEADERS} ${ORUTILS_SOURCES})


##############################################################
# Specify the include directories, target and link libraries #
##############################################################

IF(WITH_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
ELSE()
  add_definitions(-DCOMPILE_WITHOUT_CUDA)
ENDIF()

if(WITH_BACKWARDS_CPP)
  # Link agains libbfd to ensure backward-cpp can extract additional information from the binary,
  # such as source code mappings. The '-lbfd' dependency is optional, and if it is disabled, the
  # stack traces will still work, but won't show unmangled symbol names or source code snippets.
  # You may need to set BACKWARD_USE_BFD to 0 in its `hpp` and `cpp` files to avoid linker errors.
  target_link_libraries(ORUtils PUBLIC -lbfd)
  target_link_libraries(ORUtils PUBLIC backward)
endif()

