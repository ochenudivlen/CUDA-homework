#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cassert>
#include <chrono>
#include <thread>
#include <iomanip>
#include <random>

//Размер блока
#define BLOCK_SIZE 2

//Тип, который будут иметь элементы матриц
#define BASE_TYPE double

//Функция вычисления числа, которое больше a и кратно b
int toMultiple(int a, int b) 
{
    int mod = a % b;
    if (mod != 0) 
    {
        mod = b - mod;
        return a + mod;
    }
    return a;
}

__device__ void colon(BASE_TYPE* A, BASE_TYPE* I, const int Arows, const int Acols, const int n, const int m)
{
    BASE_TYPE ratio = 0;

    for (int i = 0; i < Arows; i++)
    {
        if (i != n)
        {
            if (A[n * Acols + n] != 0)
            {
                ratio = A[i * Acols + n] / A[n * Acols + n];
            }
            else
            {
                return;
            }

            A[i * Acols + threadIdx.x] -= A[n * Acols + threadIdx.x] * ratio;
            I[i * Acols + threadIdx.x] -= I[n * Acols + threadIdx.x] * ratio;
        }
    }
}

//Функция получения обратной матрицы
__global__ void inverseMatrix(BASE_TYPE* A, BASE_TYPE* I, const int Arows, const int Acols)
{
    for (int i = 0; i < Arows; i++)
    {
        colon(A, I, Arows, Acols, i, threadIdx.x);
    }

    for (int i = 0; i < Arows; i++)
    {
        BASE_TYPE t = A[i * Acols + i];

        A[i * Acols + threadIdx.x] /= t;
        I[i * Acols + threadIdx.x] /= t;
    }
}

int main()
{
    //start, stop - for Kernel time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // количество строк и столбцов матрицы
    int Arows = 1;
    int Acols = 1;

    Arows = toMultiple(Arows, BLOCK_SIZE);
    printf("Arows = %d\n", Arows);

    Acols = toMultiple(Acols, BLOCK_SIZE);
    printf("Acols = %d\n\n", Acols);

    //Проверка матрицы на квадратность
    if (Arows != Acols)
    {
        std::cout << "Matrix is not square" << std::endl;
        assert(Arows == Acols);
    }

    size_t Asize = Arows * Acols * sizeof(BASE_TYPE);

    BASE_TYPE* h_A = (BASE_TYPE*)malloc(Asize);
    BASE_TYPE* h_I = (BASE_TYPE*)malloc(Asize);
    BASE_TYPE* h_B = (BASE_TYPE*)malloc(Asize);
    BASE_TYPE* h_C = (BASE_TYPE*)malloc(Asize);

    std::random_device device;
    std::mt19937_64 engine(device());
    std::uniform_real_distribution<> distribution(0.0, 1.0);

    //Заполнение матрицы числами
    for (int i = 0; i < Arows * Acols; i++)
    {
        h_A[i] = int(distribution(engine) * 10 + 1);
    }

    for (int i = 0; i < Arows * Acols; i++)
    {
        h_I[i] = 0;
    }

    for (int i = 0; i < Arows; i++)
    {
        h_I[i * Arows + i] = 1;
    }

    std::cout << std::setprecision(3) << std::fixed;

    for (int i = 0; i < Arows; i++)
    {
        for (int j = 0; j < Acols; j++)
        {
            std::cout << h_A[i * Arows + j] << "\t";
        }

        std::cout << std::endl;
    }

    std::cout << std::endl;

    for (int i = 0; i < Arows; i++)
    {
        for (int j = 0; j < Acols; j++)
        {
            std::cout << h_I[i * Arows + j] << "\t";
        }

        std::cout << std::endl;
    }

    std::cout << std::endl;

    BASE_TYPE* d_A = NULL;
    cudaMalloc((void**)&d_A, Asize);

    BASE_TYPE* d_I = NULL;
    cudaMalloc((void**)&d_I, Asize);

    cudaMemcpy(d_A, h_A, Asize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_I, h_I, Asize, cudaMemcpyHostToDevice);

    //dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    //dim3 blocksPerGrid = 1;

    cudaEventRecord(start, 0);

    inverseMatrix<<<1, BLOCK_SIZE>>>(d_A, d_I, Arows, Acols);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float KernelTime;
    cudaEventElapsedTime(&KernelTime, start, stop);
    printf("KernelTime: %.2f milliseconds\n\n", KernelTime);

    cudaMemcpy(h_B, d_A, Asize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C, d_I, Asize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < Arows; i++)
    {
        for (int j = 0; j < Acols; j++)
        {
            std::cout << h_B[i * Arows + j] << "\t";
        }

        std::cout << std::endl;
    }

    std::cout << std::endl;

    for (int i = 0; i < Arows; i++)
    {
        for (int j = 0; j < Acols; j++)
        {
            std::cout << h_C[i * Arows + j] << "\t";
        }

        std::cout << std::endl;
    }

    std::cout << std::endl;

    free(h_A);
    free(h_I);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_I);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}