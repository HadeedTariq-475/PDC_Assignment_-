#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define DATA_SIZE 100000000  
#define RANGE 256            

void generate_data(int *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = rand() % RANGE;
    }
}

void compute_histogram_parallel_reduction(int *data, int *histogram, int size, int num_threads) {
    #pragma omp parallel for num_threads(num_threads) reduction(+:histogram[:RANGE]) schedule(static)
    for (int i = 0; i < size; i++) {
        histogram[data[i]]++;  // Directly update histogram using reduction
    }
}

int main() {
    int *data = (int *)malloc(DATA_SIZE * sizeof(int));
    int histogram[RANGE] = {0};
    int num_threads = 16;

    srand(time(NULL));
    generate_data(data, DATA_SIZE);

    double start = omp_get_wtime();
    compute_histogram_parallel_reduction(data, histogram, DATA_SIZE, num_threads);
    double end = omp_get_wtime();

    printf("Parallel Execution Time (reduction, %d threads): %f seconds\n", num_threads, end - start);

    free(data);
    return 0;
}
