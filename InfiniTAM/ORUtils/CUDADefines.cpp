#include "CUDADefines.h"

#ifdef WITH_BACKWARDS_CPP
#include "third_party/backward-cpp/backward.hpp"
#endif

namespace ORUtils {

void __cudaSafeCall(cudaError err, const char *file, const int line) {
    if (cudaSuccess != err) {

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
#else		// TODO: remove this. too verbose.
        fprintf(stderr, "\nWITH_BACKWARDS_CPP is false.\n");
#endif

        fprintf(stderr, "%s(%i) : cudaSafeCall() Runtime API error : %s.\n",
                file, line, cudaGetErrorString(err));

        exit(-1);
    }
}
}
