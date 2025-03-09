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
    int local_histograms[num_threads][RANGE]; // Local histograms for each thread

    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        for (int i = 0; i < RANGE; i++) 
            local_histograms[thread_id][i] = 0; // Initialize local histogram

        #pragma omp for schedule(static)
        for (int i = 0; i < size; i++) {
            local_histograms[thread_id][data[i]]++; // Update local histogram
        }

        #pragma omp critical
        for (int i = 0; i < RANGE; i++) {
            histogram[i] += local_histograms[thread_id][i]; // Merge local histograms
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
    printf("Parallel Execution Time (%d threads): %f seconds\n", num_threads, execution_time);

    free(data);
    return 0;
}
