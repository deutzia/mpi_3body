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

void compute_accelerations(double* accelerations)
{
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
        if (argv[5] == "-v")
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
    fprintf(stderr, "n = %d\n", n);
    int* sendcounts = malloc(sizeof(int) * world_size);
    check_mem(sendcounts);
    int* displs = malloc(sizeof(int) * world_size);
    check_mem(displs);
    for (int i = 0; i < world_size; ++i)
    {
        displs[i] = (i * n / world_size) * 3;
        sendcounts[i] = ((i+1) * n / world_size - i * n / world_size) * 3;
    }
    if (rank == 0)
    {
        for (int i = 0; i < world_size; ++i)
            fprintf(stderr, "%d %d %d\n", i, displs[i], sendcounts[i]);
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

    compute_accelerations(accelerations);
    for (int step = 1; step <= stepcount; ++step)
    {
        update_positions(positions, velocities, accelerations, sendcounts[rank], deltatime);
        compute_accelerations(new_accelerations);
        update_velocities(velocities, accelerations, new_accelerations, n, deltatime);
        double* tmp = accelerations;
        accelerations = new_accelerations;
        new_accelerations = tmp;
    }
    char buffer[100];
    sprintf(buffer, "particles-%d.out", rank);
    FILE* out = fopen(buffer, "w");
    for (int i = 0; i < sendcounts[rank] / 3; ++i)
    {
        fprintf(out, "%lf %lf %lf %lf %lf %lf\n",
                        positions[3 * i], positions[3 * i + 1],
                        positions[3 * i + 2], velocities[3 * i],
                        velocities[3 * i + 1], velocities[3 * i + 2]);
    }
    fclose(out);

    if (rank == 0)
    {
        free(positions0);
        free(velocities0);
    }
    free(positions);
    free(velocities);
    free(accelerations);
    free(sendcounts);
    free(displs);
    free(new_accelerations);
    MPI_Finalize();
    return 0;
}
