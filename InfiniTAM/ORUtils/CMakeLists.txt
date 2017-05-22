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
OPTION(WITH_BACKWARDS_CPP "Build with backwards-cpp stack trace dumping library? (Linux-only)" TRUE)
message(STATUS "backwards-cpp-enhanced stack traces? " ${WITH_BACKWARDS_CPP})


if(WITH_BACKWARDS_CPP)
  # Support 'backward-cpp,' a lean stacktrace printing library for Linux.
  add_subdirectory(third_party/backward-cpp)
  add_definitions(-DWITH_BACKWARDS_CPP)
endif()


#################################################################
# Collect the project files into common, CPU-only and CUDA-only #
#################################################################

set(ORUTILS_OBJECTS
    # TODO(andrei): Do we still need the dummy?
    Dummy.cpp
    ${ORUTILS_HEADERS}
    )

#############################
# Specify the source groups #
#############################

SOURCE_GROUP("" FILES ${ORUTILS_HEADERS} ${ORUTILS_SOURCES})

#add_library(ORUtils ${ORUTILS_OBJECTS} ${ORUTILS_SOURCES} ${BACKWARD_ENABLE})
add_library(ORUtils ${ORUTILS_OBJECTS} ${ORUTILS_SOURCES})


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
  # such as source code mappings.
  target_link_libraries(ORUtils PUBLIC -lbfd)
  target_link_libraries(ORUtils PUBLIC backward)
endif()

