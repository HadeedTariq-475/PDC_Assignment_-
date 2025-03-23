#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <string.h>

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

// 1️ Compute histogram using only `#pragma omp critical`
void compute_histogram_critical(int *data, int *histogram, int size, int num_threads) {
    int local_histograms[num_threads][RANGE];  // Local histograms
    memset(local_histograms, 0, sizeof(local_histograms));  // Zero out the array

    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();

        // First, each thread updates its local histogram
        #pragma omp for
        for (int i = 0; i < size; i++) {
            local_histograms[thread_id][data[i]]++;
        }

        // Now merge local histograms into the global histogram using critical section
        #pragma omp critical
        for (int i = 0; i < RANGE; i++) {
            histogram[i] += local_histograms[thread_id][i];
        }
    }
}

// 2️ Compute histogram using **static scheduling** + critical
void compute_histogram_static_critical(int *data, int *histogram, int size, int num_threads, int chunk_size) {
    int local_histograms[num_threads][RANGE];  // Local histograms
    memset(local_histograms, 0, sizeof(local_histograms));  // Zero out the array

    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();

        // Each thread fills its local histogram
        #pragma omp for schedule(static, chunk_size)
        for (int i = 0; i < size; i++) {
            local_histograms[thread_id][data[i]]++;
        }

        // Merge local histograms into the global histogram using critical section
        #pragma omp critical
        for (int i = 0; i < RANGE; i++) {
            histogram[i] += local_histograms[thread_id][i];
        }
    }
}

// 3 Compute histogram using **dynamic scheduling** + critical
void compute_histogram_dynamic_critical(int *data, int *histogram, int size, int num_threads, int chunk_size) {
    int local_histograms[num_threads][RANGE];  // Local histograms
    memset(local_histograms, 0, sizeof(local_histograms));  // Zero out the array
 

    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();

        // Each thread fills its local histogram
        #pragma omp for schedule(dynamic, chunk_size)
        for (int i = 0; i < size; i++) {
            local_histograms[thread_id][data[i]]++;
        }

        // Merge local histograms into the global histogram using critical section
        #pragma omp critical
        for (int i = 0; i < RANGE; i++) {
            histogram[i] += local_histograms[thread_id][i];
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

    // Define chunk sizes for static and dynamic scheduling
    int chunk_sizes[] = {32768, 65536, 131072}; // 32K, 64K, 128K
    int num_chunks = sizeof(chunk_sizes) / sizeof(chunk_sizes[0]);

    // 🔹 Test Simple Critical Section
    printf("\n--- Histogram Computation using Simple Critical ---\n");
    for (int t = 0; t < num_tests; t++) {
        int num_threads = thread_counts[t];

        printf("\nThreads: %d | Simple Critical Execution Times:\n", num_threads);
        for (int run = 0; run < NUM_RUNS; run++) {
            for (int i = 0; i < RANGE; i++) histogram[i] = 0; // Reset histogram

            double start = omp_get_wtime();
            compute_histogram_critical(data, histogram, DATA_SIZE, num_threads);
            double end = omp_get_wtime();

            printf("%f\n", end - start);
        }
    }

    // 🔹 Test Static Scheduling + Critical
    printf("\n--- Histogram Computation using Static Scheduling + Critical ---\n");
    for (int c = 0; c < num_chunks; c++) {
        int chunk_size = chunk_sizes[c];
        printf("\n-- Chunk Size: %d --\n", chunk_size);

        for (int t = 0; t < num_tests; t++) {
            int num_threads = thread_counts[t];

            printf("\nThreads: %d | Static Critical Execution Times (Chunk Size %d):\n", num_threads, chunk_size);
            for (int run = 0; run < NUM_RUNS; run++) {
                for (int i = 0; i < RANGE; i++) histogram[i] = 0; // Reset histogram

                double start = omp_get_wtime();
                compute_histogram_static_critical(data, histogram, DATA_SIZE, num_threads, chunk_size);
                double end = omp_get_wtime();

                printf("%f\n", end - start);
            }
        }
    }

    // 🔹 Test Dynamic Scheduling + Critical
    printf("\n--- Histogram Computation using Dynamic Scheduling + Critical ---\n");
    for (int c = 0; c < num_chunks; c++) {
        int chunk_size = chunk_sizes[c];
        printf("\n-- Chunk Size: %d --\n", chunk_size);

        for (int t = 0; t < num_tests; t++) {
            int num_threads = thread_counts[t];

            printf("\nThreads: %d | Dynamic Critical Execution Times (Chunk Size %d):\n", num_threads, chunk_size);
            for (int run = 0; run < NUM_RUNS; run++) {
                for (int i = 0; i < RANGE; i++) histogram[i] = 0; // Reset histogram

                double start = omp_get_wtime();
                compute_histogram_dynamic_critical(data, histogram, DATA_SIZE, num_threads, chunk_size);
                double end = omp_get_wtime();

                printf("%f\n", end - start);
            }
        }
    }

    free(data);
    return 0;
}
