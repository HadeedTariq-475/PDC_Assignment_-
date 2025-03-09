#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define DATA_SIZE 100000000  // Dataset size (we are taking relatively large dataset to clearly see the speedup)
#define RANGE 256            // Histogram range

void generate_data(int *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = rand() % RANGE; // Random numbers within range
    }
}

void compute_histogram(int *data, int *histogram, int size) {
    for (int i = 0; i < RANGE; i++) histogram[i] = 0; // Initialize histogram

    for (int i = 0; i < size; i++) {
        histogram[data[i]]++;  // Count occurrences
    }
}

int main() {

    int *data = (int *)malloc(DATA_SIZE * sizeof(int));
    int histogram[RANGE] = {0};

    srand(time(NULL));
    generate_data(data, DATA_SIZE);

    double start = omp_get_wtime();
    compute_histogram(data, histogram, DATA_SIZE);
    double end = omp_get_wtime();

    double execution_time = end - start;
    printf("Sequential Execution Time: %f seconds\n", execution_time);

    free(data);
    return 0;
}
