#include "CUDADefines.h"

#ifdef WITH_BACKWARDS_CPP
#include "third_party/backward-cpp/backward.hpp"
#endif

namespace ORUtils {

void __cudaSafeCall(cudaError err, const char *file, const int line) {
		if (cudaSuccess != err) {

#ifdef WITH_BACKWARDS_CPP
				using namespace backward;
				const int kStackTraceDepth = 32;
				// The number of top entries to skip when printing the stack trace.
        // These include two internal layers from `backward-cpp`, plus this current function.
				const int kStackTraceSkip  = 3;

				fprintf(stderr, "\nCUDA error. See stacktrace and details below:\n\n");

				// Display a helpful backtrace (with code snippets, if available).
				StackTrace st;
				st.load_here(kStackTraceDepth);

				// Disable printing out boilerplate stack frames from the stack trace
				// processing code.
				st.skip_n_firsts(kStackTraceSkip);
				Printer p;
				p.address = true;
				p.print(st);
        fprintf(stderr, "\n");
#endif

				fprintf(stderr, "%s(%i) : cudaSafeCall() Runtime API error : %s.\n",
								file, line, cudaGetErrorString(err));

				exit(-1);
		}
}
}
