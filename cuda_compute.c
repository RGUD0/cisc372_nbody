#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>

#define NUM_THREADS 256

// CUDA kernel to compute pairwise accelerations
__global__ void compute_accelerations(vector3* hPos, double* mass, vector3* dAccels) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j;
	vector3 distance;
	double magnitude_sq, magnitude, accelmag;

	if (i < NUMENTITIES) {
		for (j = 0; j < NUMENTITIES; j++) {
			if (i == j) {
				FILL_VECTOR(dAccels[i * NUMENTITIES + j], 0, 0, 0);
			} else {
				for (int k = 0; k < 3; k++) {
					distance[k] = hPos[i][k] - hPos[j][k];
				}
				magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
				magnitude = sqrt(magnitude_sq);
				accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
				FILL_VECTOR(dAccels[i * NUMENTITIES + j], 
					accelmag * distance[0] / magnitude,
					accelmag * distance[1] / magnitude,
					accelmag * distance[2] / magnitude
				);
			}
		}
	}
}

//CUDA kernel to compute acceleration sums and update velocity and position
__global__ void update_positions(vector3* hPos, vector3* hVel, double* mass, vector3* dAccels) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j, k;
	vector3 accel_sum = {0, 0, 0};

	if (i < NUMENTITIES) {
		for (j = 0; j < NUMENTITIES; j++) {
			for (k = 0; k < 3; k++0 {
				accel_sum[k] += dAccels[i * NUMENTITIES + j][k];
			}
		}
		// compute the new velocity based on the acceleration and time interval
		// compute the new position based on the velocity and time interval
		for (k = 0; k < 3; k++) {
			hVel[i][k] += accel_sum[k] * INTERVAL;
			hPos[i][k] += hVel[i][k] * INTERVAL;
		}
	}
}

void compute() {
	int num_blocks = (NUMENTITES + NUM_THREADS - 1) / NUM_THREADS;

	vector3* dAccels;
	cudaMalloc((void**)&dAccels, sizeof(vector3) * NUMENTITIES * NUMENTITIES);

	// Compute pairwise accelerations
	compute_accelerations<<<num_blocks, NUM_THREADS>>>(hPos, mass, dAccels);
	
	// Compute acceleration sums and update velocity and position
	update_positions<<<num_blocks, NUM_THREADS>>>(hPos, hVel, mass, dAccels);

	cudaFree(dAccels);
}
