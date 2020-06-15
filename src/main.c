#include <stdbool.h>
#include <errno.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <assert.h>
#include <string.h>

#include "common.h"
#include "compute_accelerations.h"

#define SIZE 100
bool verbose;
long stepcount;
double deltatime;

void update_positions(double* positions, double* velocities, double* accelerations, int n, double deltatime)
{
    for (int i = 0; i < n; ++i)
    {
        positions[i] += velocities[i] * deltatime + 0.5 * deltatime * deltatime * accelerations[i];
    }
}

void update_velocities(double* velocities, double* old_accelerations, double* new_accelerations, int n, double deltatime)
{
    for (int i = 0; i < n; ++i)
    {
        velocities[i] += 0.5 * (old_accelerations[i] + new_accelerations[i]) * deltatime;
    }
}

int main(int argc, char * argv[])
{
    if (argc < 5 || argc > 6)
    {
        printf("Incorrect usage, please run `./body3 particles_in.txt particles_out stepcount deltatime [-v]`\n");
        exit(1);
    }
    if (argc == 6)
    {
        // I assume -v can only be passed as the last argument
        if (strcmp(argv[5], "-v") == 0)
            verbose = true;
        else
        {
            printf("Incorrect usage, please run `./body3 particles_in.txt particles_out stepcount deltatime [-v]`\n");
            exit(1);
        }
    }
    stepcount = strtol(argv[3], NULL, 10);
    deltatime = strtof(argv[4], NULL);
    MPI_Init(&argc,&argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    double* positions0 = NULL;
    double* velocities0 = NULL;
    double* positions = NULL;
    double* velocities = NULL;
    double* accelerations = NULL;
    double* new_accelerations = NULL;
    int n;
    if (rank == 0)
    {
        positions0 = malloc(sizeof(double) * SIZE * 3);
        check_mem(positions0);
        velocities0 = malloc(sizeof(double) * SIZE * 3);
        check_mem(velocities0);
        int max_i = SIZE;
        FILE* file = fopen(argv[1], "r");
        if (file == NULL)
        {
            fprintf(stderr, "Error opening the file with particles (%d, %s)\n",
                    errno, strerror(errno));
            return 1;
        }
        int i = 0;
        while (fscanf(file, " %lf %lf %lf %lf %lF %lf",
                        &positions0[3 * i], &positions0[3 * i + 1],
                        &positions0[3 * i + 2], &velocities0[3 * i],
                        &velocities0[3 * i + 1], &velocities0[3 * i + 2]) == 6)
        {
            i++;
            if (i == max_i) {
                positions0 = realloc(positions0, sizeof(double) * 6 * max_i);
                check_mem(positions0);
                velocities0 = realloc(velocities0, sizeof(double) * 6 * max_i);
                check_mem(velocities0);
                max_i = 2 * max_i;
            }
        }
        fclose(file);
        n = i;
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int* sendcounts = malloc(sizeof(int) * world_size);
    check_mem(sendcounts);
    int* displs = malloc(sizeof(int) * world_size);
    check_mem(displs);
    for (int i = 0; i < world_size; ++i)
    {
        displs[i] = (i * n / world_size) * 3;
        sendcounts[i] = ((i+1) * n / world_size - i * n / world_size) * 3;
    }

    positions = malloc(sizeof(double) * sendcounts[rank]);
    check_mem(positions);
    velocities = malloc(sizeof(double) * sendcounts[rank]);
    check_mem(velocities);
    accelerations = malloc(sizeof(double) * sendcounts[rank]);
    check_mem(accelerations);
    new_accelerations = malloc(sizeof(double) * sendcounts[rank]);
    check_mem(new_accelerations);
    MPI_Scatterv(positions0, sendcounts, displs, MPI_DOUBLE,
            positions, sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(velocities0, sendcounts, displs, MPI_DOUBLE,
            velocities, sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int max_n = 0;
    for (int i = 0; i < world_size; ++i)
    {
        if (sendcounts[i] > max_n)
        {
            max_n = sendcounts[i];
        }
    }
    double* b[4] = {NULL, NULL, NULL, NULL};
    for (int i = 0; i < 4; ++i)
    {
        b[i] = malloc(sizeof(double) * max_n);
        check_mem(b[i]);
    }
    double* res[4] = {NULL, NULL, NULL, NULL};
    for (int i = 0; i < 4; ++i)
    {
        res[i] = malloc(sizeof(double) * max_n);
        check_mem(res[i]);
    }

    memset(accelerations, 0, sizeof(double) * sendcounts[rank]);
    memset(new_accelerations, 0, sizeof(double) * sendcounts[rank]);
    compute_accelerations(rank, world_size, sendcounts, max_n,
            positions, b, res, accelerations);
    for (int step = 1; step <= stepcount; ++step)
    {
        update_positions(positions, velocities, accelerations, sendcounts[rank], deltatime);

        compute_accelerations(rank, world_size, sendcounts, max_n,
                positions, b, res, new_accelerations);
        update_velocities(velocities, accelerations, new_accelerations, sendcounts[rank], deltatime);
        if (verbose)
        {
            MPI_Gatherv(positions, sendcounts[rank], MPI_DOUBLE,
                    positions0, sendcounts, displs, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
            MPI_Gatherv(velocities, sendcounts[rank], MPI_DOUBLE,
                    velocities0, sendcounts, displs, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
            if (rank == 0)
            {
                char buffer[100];
                sprintf(buffer, "particles_out_%d.txt", step);
                FILE* file = fopen(buffer, "w");
                if (file == NULL)
                {
                    fprintf(stderr, "Error opening the file to save particles (%d, %s)\n",
                            errno, strerror(errno));
                    return 1;
                }
                for (int i = 0; i < n; ++i)
                {
                    fprintf(file, "%.16lf %.16lf %.16lf %.16lf %.16lf %.16lf\n",
                                    positions0[3 * i], positions0[3 * i + 1],
                                    positions0[3 * i + 2], velocities0[3 * i],
                                    velocities0[3 * i + 1], velocities0[3 * i + 2]);
                }
                fclose(file);
            }
        }
        swap(&accelerations, &new_accelerations);
    }

    MPI_Gatherv(positions, sendcounts[rank], MPI_DOUBLE,
            positions0, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(velocities, sendcounts[rank], MPI_DOUBLE,
            velocities0, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        FILE* file = fopen(argv[2], "w");
        if (file == NULL)
        {
            fprintf(stderr, "Error opening the file to save particles (%d, %s)\n",
                    errno, strerror(errno));
            return 1;
        }
        for (int i = 0; i < n; ++i)
        {
            fprintf(file, "%.16lf %.16lf %.16lf %.16lf %.16lf %.16lf\n",
                            positions0[3 * i], positions0[3 * i + 1],
                            positions0[3 * i + 2], velocities0[3 * i],
                            velocities0[3 * i + 1], velocities0[3 * i + 2]);
        }
        fclose(file);
    }

    if (rank == 0)
    {
        free(positions0);
        free(velocities0);
    }
    free(sendcounts);
    free(displs);
    free(positions);
    free(velocities);
    free(accelerations);
    free(new_accelerations);
    for (int i = 0; i < 4; ++i)
    {
        free(b[i]);
    }
    MPI_Finalize();
    return 0;
}
