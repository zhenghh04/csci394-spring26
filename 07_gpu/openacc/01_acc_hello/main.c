#include <stdio.h>

int main(void) {
    int marker = 0;

#pragma acc parallel copy(marker)
    {
        marker = 1;
    }

    printf("OpenACC hello\n");
#ifdef _OPENACC
    printf("_OPENACC=%d\n", _OPENACC);
#else
    printf("_OPENACC not defined\n");
#endif
    printf("marker=%d\n", marker);
    if (marker == 1) {
        printf("Result: parallel region executed.\n");
    } else {
        printf("Result: region did not execute as expected.\n");
    }
    return 0;
}
