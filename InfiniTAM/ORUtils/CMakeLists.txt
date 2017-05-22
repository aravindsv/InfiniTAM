#############################
# Specify the project files #
#############################

##
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

#################################################################
# Collect the project files into common, CPU-only and CUDA-only #
#################################################################

set(ORUTILS_OBJECTS
Dummy.cpp
${ORUTILS_HEADERS}
)


# Support 'backward-cpp,' a lean stacktrace printing library for Linux.
add_subdirectory(third_party/backward-cpp)
#list(APPEND CMAKE_MODULE_PATH third_party/backward-cpp)
#list(APPEND CMAKE_PREFIX_PATH ./third_party/backward-cpp)
#message(${CMAKE_MODULE_PATH})
#list(APPEND CMAKE_MODULE_PATH :/home/andrei/work/InfiniTAM/InfiniTAM/ORUtils/third_party/backward-cpp)
#message(${CMAKE_MODULE_PATH})

#find_package(Backward)

#############################
# Specify the source groups #
#############################

SOURCE_GROUP("" FILES ${ORUTILS_HEADERS})

add_library(ORUtils ${ORUTILS_OBJECTS})

#target_link_libraries(ORUtils PUBLIC Backward::Backward)

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
