#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define DATA_SIZE 100000000  // Large dataset
#define RANGE 256            // Histogram range

void generate_data(int *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = rand() % RANGE; // Random numbers within range
    }
}

void compute_histogram_parallel(int *data, int *histogram, int size, int num_threads) {
    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp for
        for (int i = 0; i < size; i++) {
            #pragma omp atomic
            histogram[data[i]]++; // Atomic update to avoid race conditions
        }
    }
}

int main() {
    int *data = (int *)malloc(DATA_SIZE * sizeof(int));
    int histogram[RANGE] = {0};
    int num_threads = 16; // Set number of threads

    srand(time(NULL));
    generate_data(data, DATA_SIZE);

    double start = omp_get_wtime();
    compute_histogram_parallel(data, histogram, DATA_SIZE, num_threads);
    double end = omp_get_wtime();

    double execution_time = end - start;
    printf("Parallel Execution Time (%d threads, atomic): %f seconds\n", num_threads, execution_time);

    free(data);
    return 0;
}

