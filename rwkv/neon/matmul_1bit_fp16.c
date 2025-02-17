// Compile command: 
/*
    aarch64-linux-gnu-gcc -o matmul_1bit_fp16 matmul_1bit_fp16.c \
    -march=armv8.2-a+fp16 -O3
    https://chatgpt.com/share/6719a1ee-9f64-8004-82fb-aa13aa6c5242

    obs: neon opt much better than no opt. however seems still 2x slower than 
    torch fp32 version. multi threading no much benefit for 1024x1024x256. 
    even for 2048x2048x512    (to profile later)
    
    results: see end of file
*/    


#include <time.h>
#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <omp.h>     // For OpenMP

void matmul_1bit_fp16(const uint8_t* weight_1bit, const float16_t* input_fp16, float16_t* output_fp16, 
                      const float16_t* bias_fp16, int M, int N, int K, float16_t scale) {
    // M: Number of rows of output
    // N: Number of columns of output (and input)
    // K: Number of columns of weight (and rows of input)
    // weight_1bit: Packed 1-bit weights, size: (M * K + 7) / 8
    // input_fp16: Input matrix, size: (K * N)
    // output_fp16: Output matrix, size: (M * N)
    // bias_fp16: Bias array, size: M

    // Print all values in input_fp16
    if (M<128 && N<128 && K<128) {
        printf("Input_fp16 values:\n");
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < N; ++j) {
                printf("%f ", (float)input_fp16[i * N + j]);
            }
            printf("\n");
        }
        printf("------------ \n");
    }

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float16_t sum = bias_fp16[m];

            for (int k = 0; k < K; ++k) {
                // Extract the bit from weight_1bit
                int bit_index = m * K + k;
                int byte_index = bit_index / 8;
                int bit_offset = bit_index % 8;
                uint8_t bit = (weight_1bit[byte_index] >> bit_offset) & 1;

                // Dequantize the bit to fp16 (-scale or +scale)
                float16_t weight_fp16 = bit ? scale : -scale;

                // Load input value
                float16_t input_value = input_fp16[k * N + n];

                if (isnan(input_value)) {
                    // printf("WARNING k: %d, n: %d, k*N+n: %d\n", k, n, k * N + n);
                    // printf("input_fp16[%d][%d] = %f\n", k, n, (float)input_fp16[k * N + n]);

                    printf("------------ Input_fp16 values:\n");
                    for (int i = 0; i < K; ++i) {
                        for (int j = 0; j < N; ++j) {
                            printf("%f ", (float)input_fp16[i * N + j]);
                        }
                        printf("\n");
                    }
                    printf("------------ \n");
                }

                // Accumulate the product
                sum += weight_fp16 * input_value;
            }

            // Store the result in the output matrix
            output_fp16[m * N + n] = sum;

            // Check if the result is NaN and print a warning
            if (isnan(output_fp16[m * N + n])) {
                printf("Warning: output_fp16[%d][%d] is NaN\n", m, n);
            }
        }
    }
}

// Optimized version using NEON fp16 intrinsics
void matmul_1bit_fp16_neon_v1(const uint8_t* weight_1bit, const float16_t* input_fp16, float16_t* output_fp16, 
                           const float16_t* bias_fp16, int M, int N, int K, float16_t scale) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; n += 4) {
            // Initialize output accumulator with bias
            float16x4_t sum_vec = vdup_n_f16(bias_fp16[m]);

            for (int k = 0; k < K; ++k) {
                // Extract the bit from weight_1bit
                int bit_index = m * K + k;
                int byte_index = bit_index / 8;
                int bit_offset = bit_index % 8;
                uint8_t bit = (weight_1bit[byte_index] >> bit_offset) & 1;

                // Dequantize the bit to fp16 (-scale or +scale)
                float16_t weight_fp16 = bit ? scale : -scale;
                float16x4_t weight_vec = vdup_n_f16(weight_fp16);

                // Load 4 input values
                float16x4_t input_vec = vld1_f16(&input_fp16[k * N + n]);

                // Multiply and accumulate
                sum_vec = vfma_f16(sum_vec, weight_vec, input_vec);
            }

            // Store the result in the output matrix
            vst1_f16(&output_fp16[m * N + n], sum_vec);
        }
    }
}

// Optimized version using NEON fp16 intrinsics with NEON-based multiple-bit extraction
// works
void matmul_1bit_fp16_neon_v2(const uint8_t* weight_1bit, const float16_t* input_fp16, float16_t* output_fp16,
                           const float16_t* bias_fp16, int M, int N, int K, float16_t scale) {
    float16x4_t scale_pos_vec = vdup_n_f16(scale);
    float16x4_t scale_neg_vec = vdup_n_f16(-scale);

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; n += 4) {
            // Initialize sum vector with bias
            float16x4_t sum_vec = vdup_n_f16(bias_fp16[m]);

            for (int k = 0; k < K; ++k) {
                // Extract the bit from weight_1bit
                int bit_index = m * K + k;
                int byte_index = bit_index / 8;
                int bit_offset = bit_index % 8;
                uint8_t bit = (weight_1bit[byte_index] >> bit_offset) & 1;

                // Dequantize the bit to fp16 (-scale or +scale)
                float16x4_t weight_vec = bit ? scale_pos_vec : scale_neg_vec;

                // Load input vector (4 elements)
                float16x4_t input_vec = vld1_f16(&input_fp16[k * N + n]);

                // Multiply and accumulate
                sum_vec = vfma_f16(sum_vec, weight_vec, input_vec);
            }

            // Store the result
            vst1_f16(&output_fp16[m * N + n], sum_vec);
        }
    }
}

// v3, single thr, no openmp
void matmul_1bit_fp16_neon_v3(
    const uint8_t* weight_1bit,
    const float16_t* input_fp16,
    float16_t* output_fp16,
    const float16_t* bias_fp16,
    int M,
    int N,
    int K,
    float scale) {
    // Ensure N is a multiple of 8
    assert(N % 8 == 0 && "Number of columns N must be a multiple of 8.");

    // Define NEON vectors for +scale and -scale
    float16x8_t scale_pos_vec = vdupq_n_f16((float16_t)scale);
    float16x8_t scale_neg_vec = vdupq_n_f16((float16_t)(-scale));
    
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; n += 8) { // Process 8 columns at a time
            // Initialize sum vector with bias
            float16x8_t sum_vec = vdupq_n_f16(bias_fp16[m]);

            for (int k = 0; k < K; ++k) {
                // Calculate bit index
                int bit_index = m * K + k;
                int byte_index = bit_index / 8;
                int bit_offset = bit_index % 8;

                // Load the byte containing the bit
                uint8_t byte = weight_1bit[byte_index];

                // Extract the bit: 1 or 0
                uint8_t bit = (byte >> bit_offset) & 1;

                // Create a 16-bit mask: all bits set if bit is 1, else 0
                uint16_t mask_val = bit ? 0xFFFF : 0x0000;

                // Duplicate the mask across all 16-bit lanes
                uint16x8_t mask_vec = vdupq_n_u16(mask_val);

                // Select scale based on the mask
                float16x8_t weight_vec = vbslq_f16(mask_vec, scale_pos_vec, scale_neg_vec);

                // Load 8 input values
                float16x8_t input_vec = vld1q_f16(&input_fp16[k * N + n]);

                // Multiply and accumulate: sum += weight * input
                sum_vec = vfmaq_f16(sum_vec, weight_vec, input_vec);
            }

            // Store the result
            vst1q_f16(&output_fp16[m * N + n], sum_vec);
        }
    }
}


// v4, with omp
void matmul_1bit_fp16_neon(
    const uint8_t* weight_1bit,
    const float16_t* input_fp16,
    float16_t* output_fp16,
    const float16_t* bias_fp16,
    int M,
    int N,
    int K,
    float scale
) {
    // Ensure N is a multiple of 8
    assert(N % 8 == 0 && "Number of columns N must be a multiple of 8.");

    // Define NEON vectors for +scale and -scale
    float16x8_t scale_pos_vec = vdupq_n_f16((float16_t)scale);
    float16x8_t scale_neg_vec = vdupq_n_f16((float16_t)(-scale));

    // Parallelize the outer loop over rows using OpenMP
    #pragma omp parallel for
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; n += 8) { // Process 8 columns at a time
            // Initialize sum vector with bias
            float16x8_t sum_vec = vdupq_n_f16(bias_fp16[m]);

            for (int k = 0; k < K; ++k) {
                // Calculate bit index
                int bit_index = m * K + k;
                int byte_index = bit_index / 8;
                int bit_offset = bit_index % 8;

                // Load the byte containing the bit
                uint8_t byte = weight_1bit[byte_index];

                // Extract the bit: 1 or 0
                uint8_t bit = (byte >> bit_offset) & 1;

                // Create a 16-bit mask: all bits set if bit is 1, else 0
                uint16_t mask_val = bit ? 0xFFFF : 0x0000;

                // Duplicate the mask across all 16-bit lanes
                uint16x8_t mask_vec = vdupq_n_u16(mask_val);

                // Select scale based on the mask
                float16x8_t weight_vec = vbslq_f16(mask_vec, scale_pos_vec, scale_neg_vec);

                // Load 8 input values
                float16x8_t input_vec = vld1q_f16(&input_fp16[k * N + n]);

                // Multiply and accumulate: sum += weight * input
                sum_vec = vfmaq_f16(sum_vec, weight_vec, input_vec);
            }

            // Store the result
            vst1q_f16(&output_fp16[m * N + n], sum_vec);
        }
    }
}


// vfma: Floating-point fused Multiply-Add to accumulator (vector)
// Simple test code
int main() {
    // Define matrix dimensions
    // int M = 2;  // Number of rows of output
    // int N = 8;  // Number of columns of output (and input)
    // int K = 8;  // Number of columns of weight (and rows of input)

    // int M = 1024;  // Number of rows of output
    // int N = 1024;  // Number of columns of output (and input)
    // int K = 256;  // Number of columns of weight (and rows of input)

    int M = 2048;  // Number of rows of output
    int N = 2048;  // Number of columns of output (and input)
    int K = 512;  // Number of columns of weight (and rows of input)

    // Define scale factor for dequantization
    float16_t scale = 0.5f;

    // Define input matrices
    uint8_t* weight_1bit = (uint8_t*)malloc((M * K + 7) / 8 * sizeof(uint8_t));  // Packed 1-bit weights
    float16_t* input_fp16 = (float16_t*)malloc(K * N * sizeof(float16_t));
    float16_t* bias_fp16 = (float16_t*)malloc(M * sizeof(float16_t));

    if (weight_1bit == NULL || input_fp16 == NULL || bias_fp16 == NULL) {
        printf("Memory allocation failed\n");
        return -1;
    }

    // Initialize weight_1bit, input_fp16, and bias_fp16
    for (int i = 0; i < (M * K + 7) / 8; ++i) {
        weight_1bit[i] = (uint8_t)(i % 256);  // Initialize with values from 0 to 255
    }
    // Initialize input_fp16 and bias_fp16
    for (int i = 0; i < K * N; ++i) {
        input_fp16[i] = (float16_t)(i % 8 + 1);  // Initialize with values from 1.0 to 8.0
    }
    for (int i = 0; i < M; ++i) {
        bias_fp16[i] = (float16_t)(i % 2 == 0 ? 1.0f : -1.0f);  // Bias alternating between 1.0 and -1.0
    }


    // Output matrices
    float16_t* output_fp16 = (float16_t*)malloc(M * N * sizeof(float16_t));        // M x N for standard version
    float16_t* output_fp16_neon = (float16_t*)malloc(M * N * sizeof(float16_t));   // M x N for NEON version

    if (output_fp16 == NULL || output_fp16_neon == NULL) {
        printf("Memory allocation failed\n");
        return -1;
    }
    
    int num_threads = omp_get_max_threads();
    if (N<500 && M<500) 
        num_threads = 1;
    else if (N<1000 && M<1000) 
        num_threads = 2;

    // Measure execution time for standard version
    clock_t start = clock();
    matmul_1bit_fp16(weight_1bit, input_fp16, output_fp16, bias_fp16, M, N, K, scale);
    clock_t end = clock();
    double time_spent_standard = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Standard version execution time: %f seconds\n", time_spent_standard);

    // Measure execution time for NEON version
    start = clock();    
    matmul_1bit_fp16_neon_v3(weight_1bit, input_fp16, output_fp16_neon, 
        bias_fp16, M, N, K, scale);
    end = clock();
    double time_spent_neon = (double)(end - start) / CLOCKS_PER_SEC;
    printf("NEON version execution time: %f seconds\n", time_spent_neon);

    if (M<128 && N<128 && K<128) {
        // Print and compare the output matrices
        printf("Standard Version Output:\n");
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                printf("%f ", (float)output_fp16[m * N + n]);
            }
            printf("\n");
        }

        printf("\nNEON Version Output:\n");
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                printf("%f ", (float)output_fp16_neon[m * N + n]);
            }
            printf("\n");
        }


        // Compare the results
        int match = 1;
        for (int i = 0; i < M * N; ++i) {
            if (output_fp16[i] != output_fp16_neon[i]) {
                printf("Mismatch at index %d: Standard = %f, NEON = %f\n", i, (float)output_fp16[i], (float)output_fp16_neon[i]);
                match = 0;
            }
        }

        if (match) {
            printf("\nResults match!\n");
        } else {
            printf("\nResults do not match!\n");
        }
    }

    return 0;
}

/*

rpi5

    int M = 1024;  // Number of rows of output
    int N = 1024;  // Number of columns of output (and input)
    int K = 256;  // Number of columns of weight (and rows of input)


Standard version execution time: 1.276969 seconds
NEON version execution time: 0.165859 seconds       // matmul_1bit_fp16_neon_v2
NEON version execution time: 0.077954 seconds       // matmul_1bit_fp16_neon_v3
Results match!


    int M = 2048;  // Number of rows of output
    int N = 2048;  // Number of columns of output (and input)
    int K = 512;  // Number of columns of weight (and rows of input)

std: >10 sec
neon v3 with openmp static schedule ... 1.6 sec
neon v3 w/o openmp .... 1.3 sec (better than no openmp??) why
*/