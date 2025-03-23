#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define DATA_SIZE 100000000  // Dataset size (large dataset to see performance clearly)
#define RANGE 256            // Histogram range
#define RUNS 10              // Number of times to repeat execution

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

    printf("Sequential Execution Times for %d Runs:\n", RUNS);

    for (int run = 0; run < RUNS; run++) {
        // Reset histogram before each run
        for (int i = 0; i < RANGE; i++) histogram[i] = 0;

        // Measure execution time
        double start = omp_get_wtime();
        compute_histogram(data, histogram, DATA_SIZE);
        double end = omp_get_wtime();

        double execution_time = end - start;
        printf("Run %d: %f seconds\n", run + 1, execution_time);
    }

    free(data);
    return 0;
}
