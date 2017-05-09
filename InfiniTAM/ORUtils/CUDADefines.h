// Copyright 2014-2015 Isis Innovation Limited and the authors of InfiniTAM

#pragma once

#ifdef WITH_BACKWARDS_CPP
#include "third_party/backward-cpp/backward.hpp"
#endif

#ifndef COMPILE_WITHOUT_CUDA

#if (!defined USING_CMAKE) && (defined _MSC_VER)
#pragma comment( lib, "cuda.lib" )
#pragma comment( lib, "cudart.lib" )
#pragma comment( lib, "cublas.lib" )
#pragma comment( lib, "cufft.lib" )
#endif

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  include <windows.h>
#endif

#ifndef ORcudaSafeCall
#define ORcudaSafeCall(err) ORUtils::__cudaSafeCall(err, __FILE__, __LINE__)

namespace ORUtils {

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) {

#ifdef WITH_BACKWARDS_CPP
	    using namespace backward;
	    fprintf(stderr, "\nCUDA error. See stacktrace and details below:\n\n");
	    // Display a helpful backtrace (with code snippets, if available).
	    StackTrace st;
	    st.load_here(32);
	    // Disable printing out boilerplate stack frames from the stack trace
	    // processing code.
	    st.skip_n_firsts(3);
			Printer p;
			p.address = true;
	    p.print(st);
#endif

	    fprintf(stderr, "%s(%i) : cudaSafeCall() Runtime API error : %s.\n",
	           file, line, cudaGetErrorString(err) );

			exit(-1);
    }
}

}

#endif    // ifndef ORcudaSafeCall
#endif    // ifndef COMPILE_WITHOUT_CUDA

