#include "compute_accelerations.h"

#include <stdbool.h>
#include <mpi.h>
#include <stdio.h>
#include <string.h>

#include "common.h"

#include <math.h>

const double REPS = 10.0e-10;
const double HEPS = 4.69041575982343e-08;

const int T[3][3] = {{1, 0, 0}, {0, 1, 0}, {0,0, 1}};

int prv(int i, int p)
{
    return (i + p - 1) % p;
}

int nxt(int i, int p)
{
    return (i + 1) % p;
}

double compute_vijk(double r0, double r1, double r2)
{
    double prod = r0 * r1 * r2;
    double result =  1. / (prod * prod * prod) + 3 * (-r0*r0 + r1*r1 + r2*r2) * (r0*r0 - r1*r1 + r2*r2) * (r0*r0 + r1*r1 - r2*r2) / (8. * prod * prod * prod * prod * prod);
    return result;
}

double distance(double x1, double y1, double z1,
                double x2, double y2, double z2)
{
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
}

void compute(double** b, double** res,
        int i0, int i1, int i2,
        int size0, int size1, int size2)
{
    double r0, r1, r2, nval0, nval1, nval2, h;
    int it[3] = {0, 0, 0};
    int i[3] = {i0, i1, i2};
    bool f1 = (memcmp(b[i0], b[i1], sizeof(double) * size0) == 0);
    bool f2 = (memcmp(b[i1], b[i2], sizeof(double) * size1) == 0);
    bool f3 = (memcmp(b[i0], b[i2], sizeof(double) * size0) == 0);

    for (it[0] = 0; it[0] < size0; it[0] += 3)
    {
        for (it[1] = 0; it[1] < size1; it[1] += 3)
        {
            if (f1 && it[0] <= it[1])
            {
                break;
            }
            for (it[2] = 0; it[2] < size2; it[2] += 3)
            {
                if (f2 && it[1] <= it[2])
                {
                    break;
                }
                if (f3 && it[0] <= it[2])
                {
                    break;
                }
                for (int particle0 = 0; particle0 < 3; ++particle0)
                {
                    int particle1 = nxt(particle0, 3);
                    int particle2 = nxt(particle1, 3);
                    int p0 = i[particle0];
                    int p1 = i[particle1];
                    int p2 = i[particle2];
                    r0 = distance(b[p1][it[particle1]], b[p1][it[particle1]+1], b[p1][it[particle1]+2],
                                  b[p2][it[particle2]], b[p2][it[particle2]+1], b[p2][it[particle2]+2]);
                    r1 = distance(b[p0][it[particle0]], b[p0][it[particle0]+1], b[p0][it[particle0]+2],
                                  b[p2][it[particle2]], b[p2][it[particle2]+1], b[p2][it[particle2]+2]);
                    r2 = distance(b[p1][it[particle1]], b[p1][it[particle1]+1], b[p1][it[particle1]+2],
                                  b[p0][it[particle0]], b[p0][it[particle0]+1], b[p0][it[particle0]+2]);

                    if (r0 == 0 || r1 == 0 || r2 == 0)
                    {
                        // this is the case where two buffers with the same contents but
                        // different numbers meet (therefore exact 0, not close-to-zero
                        // value)
                        continue;
                    }
                    if (r0 < REPS)
                    {
                        r0 = REPS;
                    }

                    for (int dim = 0; dim < 3; ++dim)
                    {
                        h = b[p0][it[particle0] + dim] * HEPS;
                        nval0 = b[p0][it[particle0]] + h * T[dim][0];
                        nval1 = b[p0][it[particle0] + 1] + h * T[dim][1];
                        nval2 = b[p0][it[particle0] + 2] + h * T[dim][2];
                        r1 = distance(b[p1][it[particle1]], b[p1][it[particle1] + 1], b[p1][it[particle1] + 2],
                                      nval0, nval1, nval2);
                        if (r1 < REPS)
                        {
                            r1= REPS;
                        }
                        r2 = distance(b[p2][it[particle2]], b[p2][it[particle2] + 1], b[p2][it[particle2] + 2],
                                nval0, nval1, nval2);
                        if (r2 < REPS)
                        {
                            r2 = REPS;
                        }

                        double tmp1 = compute_vijk(r0, r1, r2);

                        nval0 = b[p0][it[particle0]] - h * T[dim][0];
                        nval1 = b[p0][it[particle0] + 1] - h * T[dim][1];
                        nval2 = b[p0][it[particle0] + 2] - h * T[dim][2];
                        r1 = distance(b[p1][it[particle1]], b[p1][it[particle1] + 1], b[p1][it[particle1] + 2],
                                      nval0, nval1, nval2);
                        r2 = distance(b[p2][it[particle2]], b[p2][it[particle2] + 1], b[p2][it[particle2] + 2],
                                nval0, nval1, nval2);
                        res[p0][it[particle0] + dim] -= 2 * (tmp1 - compute_vijk(r0, r1, r2));
                    }

                }
            }
        }
    }
}

void compute_accelerations(int rank, int world_size, int* sendcounts, int bsize,
        double* positions, double** b, double** res, double* accelerations)
{
    MPI_Status status;
    memcpy(b[1], positions, sendcounts[rank] * sizeof(double));
    for (int i = 0; i < 4; ++i)
    {
        memset(res[i], 0, sizeof(double) * bsize);
    }
    const int left = prv(rank, world_size);
    const int right = nxt(rank, world_size);
    MPI_Sendrecv(b[1], sendcounts[rank], MPI_DOUBLE, right, 0,
                 b[0], sendcounts[left], MPI_DOUBLE, left, 0,
                 MPI_COMM_WORLD, &status);
    MPI_Sendrecv(b[1], sendcounts[rank], MPI_DOUBLE, left, 0,
                 b[2], sendcounts[right], MPI_DOUBLE, right, 0,
                 MPI_COMM_WORLD, &status);

    // whose buffers I currently have
    int my_buffers[3] = {left, rank, right};
    // where are my buffers
    int positions_mine[3] = {right, rank, left};
    int i = 0;

    for (int s = world_size - 3; s >= 0; s -= 3)
    {
        for (int move = 0; move < s; ++move)
        {
            if (move != 0 || s != world_size - 3)
            {
                // shift b[i]
                my_buffers[i] = prv(my_buffers[i], world_size);
                positions_mine[i] = nxt(positions_mine[i], world_size);
                MPI_Sendrecv(b[i], bsize, MPI_DOUBLE, right,0,
                             b[3], bsize, MPI_DOUBLE, left, 0,
                             MPI_COMM_WORLD, &status);
                swap(&b[i], &b[3]);
                MPI_Sendrecv(res[i], bsize, MPI_DOUBLE, right, 0,
                             res[3], bsize, MPI_DOUBLE, left, 0,
                             MPI_COMM_WORLD, &status);
                swap(&res[i], &res[3]);
            }
            else
            {
                compute(b, res, 1, 1, 1, sendcounts[my_buffers[1]], sendcounts[my_buffers[1]], sendcounts[my_buffers[1]]);
                compute(b, res, 1, 1, 2, sendcounts[my_buffers[1]], sendcounts[my_buffers[1]], sendcounts[my_buffers[2]]);
                compute(b, res, 0, 0, 2, sendcounts[my_buffers[0]], sendcounts[my_buffers[0]], sendcounts[my_buffers[2]]);
            }
            if (s == world_size - 3)
            {
                compute(b, res, 0, 1, 1, sendcounts[my_buffers[0]], sendcounts[my_buffers[1]], sendcounts[my_buffers[1]]);
            }
            compute(b, res, 0, 1, 2, sendcounts[my_buffers[0]], sendcounts[my_buffers[1]], sendcounts[my_buffers[2]]);
        }
        i = (i + 1) % 3;
    }
    if (world_size % 3 == 0)
    {
        if (world_size % 3 == 0)
        {
            i = prv(i, 3);
        }

        // shift b[i]
        my_buffers[i] = prv(my_buffers[i], world_size);
        positions_mine[i] = nxt(positions_mine[i], world_size);
        MPI_Sendrecv(b[i], bsize, MPI_DOUBLE, right,0,
                     b[3], bsize, MPI_DOUBLE, left, 0,
                     MPI_COMM_WORLD, &status);
        swap(&b[i], &b[3]);
        MPI_Sendrecv(res[i], bsize, MPI_DOUBLE, right, 0,
                     res[3], bsize, MPI_DOUBLE, left, 0,
                     MPI_COMM_WORLD, &status);
        swap(&res[i], &res[3]);

        if ((rank / (world_size / 3)) == 0)
        {
            compute(b, res, 0, 1, 2, sendcounts[my_buffers[0]], sendcounts[my_buffers[1]], sendcounts[my_buffers[2]]);
        }
    }

    // Send particles in each buffer back to the owner processor
    for (int i = 0; i < 3; ++i)
    {
        MPI_Sendrecv(res[i], bsize, MPI_DOUBLE, my_buffers[i], 0,
                     res[3], bsize, MPI_DOUBLE, positions_mine[i], 0,
                     MPI_COMM_WORLD, &status);
        swap(&res[i], &res[3]);
    }

    // Sum the forces over all 3 copies
    for (int i = 0; i < sendcounts[rank]; i++)
    {
        res[0][i] += res[1][i] + res[2][i];
    }

    for (int i = 0; i < sendcounts[rank]; ++i)
    {
        double h = HEPS * positions[i];
        volatile double d1 = positions[i] + h;
        volatile double d2 = positions[i] - h;
        accelerations[i] = res[0][i] / (d1 - d2);
    }
}

