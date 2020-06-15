#include "common.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void check_mem(void* ptr)
{
    if (!ptr)
    {
        fprintf(stderr, "Error allocating memory (%d, %s)\n",
                errno, strerror(errno));
        exit(1);
    }
}

void swap(double** a, double** b)
{
    double* tmp = *a;
    *a = *b;
    *b = tmp;
}
