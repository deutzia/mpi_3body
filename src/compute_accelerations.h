#ifndef __COMPUTE_ACCELERATIONS_H__
#define __COMPUTE_ACCELERATIONS_H__

#include <mpi.h>

void compute_accelerations(int rank, int world_size, int* sendcounts, int bsize,
        double* positions, double** b, double** res, double* accelerations);

#endif // __COMPUTE_ACCELERATIONS_H__

