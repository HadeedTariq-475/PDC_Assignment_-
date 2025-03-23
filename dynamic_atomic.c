#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define DATA_SIZE 100000000  // Large dataset
#define RANGE 256            // Histogram range
#define NUM_RUNS 10          // Number of runs per configuration

// Function to generate random data in parallel
void generate_data(int *data, int size) {
    #pragma omp parallel
    {
        unsigned int seed = time(NULL) ^ omp_get_thread_num(); // Unique seed for each thread
        #pragma omp for
        for (int i = 0; i < size; i++) {
            data[i] = rand_r(&seed) % RANGE; // Thread-safe random number
        }
    }
}

// Function to compute histogram using dynamic scheduling with chunk sizes
void compute_histogram_dynamic(int *data, int *histogram, int size, int num_threads, int chunk_size) {
    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp for schedule(dynamic, chunk_size)
        for (int i = 0; i < size; i++) {
            #pragma omp atomic
            histogram[data[i]]++;
        }
    }
}

int main() {
    int *data = (int *)malloc(DATA_SIZE * sizeof(int));
    int histogram[RANGE];

    srand(time(NULL));
    generate_data(data, DATA_SIZE);

    // Define thread counts to test
    int thread_counts[] = {2, 4, 6, 8, 12, 16};
    int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);

    // Define dynamic scheduling chunk sizes
    int chunk_sizes[] = {32768, 65536, 131072}; // 32K, 64K, 128K
    int num_chunks = sizeof(chunk_sizes) / sizeof(chunk_sizes[0]);

    // Run tests for dynamic scheduling with different chunk sizes
    printf("\n--- Histogram Computation using Dynamic Scheduling ---\n");
    for (int c = 0; c < num_chunks; c++) {
        int chunk_size = chunk_sizes[c];
        printf("\n-- Chunk Size: %d --\n", chunk_size);

        for (int t = 0; t < num_tests; t++) {
            int num_threads = thread_counts[t];

            printf("\nThreads: %d | Dynamic Execution Times (Chunk Size %d):\n", num_threads, chunk_size);
            for (int run = 0; run < NUM_RUNS; run++) {
                for (int i = 0; i < RANGE; i++) histogram[i] = 0; // Reset histogram

                double start = omp_get_wtime();
                compute_histogram_dynamic(data, histogram, DATA_SIZE, num_threads, chunk_size);
                double end = omp_get_wtime();

                printf("%f\n", end - start); // Print each execution time
            }
        }
    }

    free(data);
    return 0;
}
