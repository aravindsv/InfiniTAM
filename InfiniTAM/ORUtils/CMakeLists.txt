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

IF(WITH_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
ELSE()
  add_definitions(-DCOMPILE_WITHOUT_CUDA)
ENDIF()

# Support 'backward-cpp,' a lean stacktrace printing library for Linux.
add_subdirectory(third_party/backward-cpp)
add_definitions(-DWITH_BACKWARDS_CPP)

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

SOURCE_GROUP("" FILES ${ORUTILS_HEADERS} CUDADefines.cpp)

add_library(ORUtils ${ORUTILS_OBJECTS} CUDADefines.cpp ${BACKWARD_ENABLE})

# TODO(andrei): Clean this up if it works.

#list(APPEND CMAKE_MODULE_PATH third_party/backward-cpp)

#target_link_libraries(ORUtils PUBLIC backward)
#target_link_libraries(ORUtils PUBLIC libbfd backward)
#target_link_libraries(ORUtils libbfd)
target_link_libraries(ORUtils PUBLIC -lbfd)
#target_link_libraries(ORUtils PUBLIC lbfd)
target_link_libraries(ORUtils PUBLIC bfd)
#target_link_libraries(ORUtils PUBLIC libbfd)
target_link_libraries(ORUtils PUBLIC backward)

##############################################################
# Specify the include directories, target and link libraries #
##############################################################

IF(WITH_CUDA)
#  include_directories(${CUDA_INCLUDE_DIRS})
#  cuda_add_library(ITMLib
#	${ITMLIB_CPU_OBJECTS}
#	${ITMLIB_CUDA_OBJECTS}
#	${ITMLIB_COMMON_OBJECTS}
#	OPTIONS -gencode arch=compute_11,code=compute_11 -gencode arch=compute_30,code=compute_30)
#  target_link_libraries(ITMLib ${CUDA_LIBRARY})
ELSE()
#  add_definitions(-DCOMPILE_WITHOUT_CUDA)
#  add_library(ITMLib ${ITMLIB_CPU_OBJECTS} ${ITMLIB_COMMON_OBJECTS})
ENDIF()

#target_link_libraries(ITMLib Utils)
