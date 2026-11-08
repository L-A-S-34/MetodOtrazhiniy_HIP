#pragma once
#include "msvc_defines.h"
#include <hip/hip_runtime.h>
#include <cmath>
#include <iostream>

template <typename T>
__device__ __forceinline__ void KahanStepGPU(T value, T& sum, T& err)
{
	T y = value - err;
	T t = sum + y;
	err = (t - sum) - y;
	sum = t;
}

template <typename T>
void KahanStepCPU(T value, T& sum, T& err)
{
	T y = value - err;
	T t = sum + y;
	err = (t - sum) - y;
	sum = t;
}

template <typename T>
__global__ void KahanWorker(T* d_A, const T* d_v, T a, int n, int k) //воркер для потока карты, каждый поток обрабатывает свой столбец
{
	// Вычисляем индекс столбца j для данного потока GPU
	int j = blockIdx.x * blockDim.x + threadIdx.x + (k + 1);

	// Защита от вылета за границы матрицы
	if (j < n)
	{
		//скалярное произведение v^t * A_j с алгоритмом Кэхэна
		T product_A = T(0);
		T err_prod = T(0);

		KahanStepGPU(d_v[0] * d_A[k * n + j], product_A, err_prod); //первый элемент

		for (int i = k + 1; i < n; i++)
		{
			KahanStepGPU(d_v[i - k] * d_A[i * n + j], product_A, err_prod);
		}

		//меняем элементы j-го столбца
		d_A[k * n + j] -= d_v[0] * a * product_A;
		for (int i = k + 1; i < n; i++)
		{
			d_A[i * n + j] -= a * d_v[i - k] * product_A;
		}
	}
}

template <typename T>
int SolveHIPKahan(T* A, const int n, T* b, T* x) //координирующая функция ядер карты
{
	//выбор карты
	hipSetDevice(0);

	T* d_A = nullptr; //матрица A на GPU
	T* d_v = nullptr; //вектор v на GPU

	hipMalloc(&d_A, n * n * sizeof(T)); //выделяет видеопамять под матрицу A
	hipMemcpy(d_A, A, n * n * sizeof(T), hipMemcpyHostToDevice); //копируем матрицу A в видеопамять из ОЗУ (куда, откуда, размер в байтах, спец.флаг)
	//hipMemcpyHostToDevice - из хоста в устройство
	//hipMemcpyDeviceToHost - из устройства в хост

	T err = T(0);
	for (int k = 0; k < n - 1; k++)
	{
		T norm_x = T(0); // норма k - го столбца, считаем на процессоре
		err = T(0); 

		for (int i = k; i < n; i++)
		{
			KahanStepCPU(A[i * n + k] * A[i * n + k], norm_x, err);
		}
		norm_x = std::sqrt(norm_x);

		if (norm_x > T(0))
		{
			T sign = (A[k * n + k] >= T(0)) ? T(1) : T(-1); //ищем вектор v и его норму, считаем на процессоре

			T* v = new T[n - k];
			v[0] = A[k * n + k] + sign * norm_x;

			T norm_v_2 = T(0);
			T err_v = T(0);

			KahanStepCPU(v[0] * v[0], norm_v_2, err_v);
			for (int i = k+1; i < n; i++)
			{
				v[i - k] = A[i * n + k];
				KahanStepCPU(v[i - k] * v[i - k], norm_v_2, err_v);
			}

			if (norm_v_2 > T(0))
			{
				T a = T(2) / norm_v_2;

				// Пересчет вектора правой части b на CPU
				T product_b = v[0] * b[k];
				T err_b = T(0);
				for (int i = k + 1; i < n; i++)
				{
					KahanStepCPU(v[i - k] * b[i], product_b, err_b);
				}

				b[k] -= a * v[0] * product_b;
				for (int i = k + 1; i < n; i++)
				{
					b[i] -= a * v[i - k] * product_b;
				}

				hipMalloc(&d_v, (n - k) * sizeof(T)); //выделяем видеопамять под вектор v 
				hipMemcpy(d_v, v, (n - k) * sizeof(T), hipMemcpyHostToDevice); // копируем v в видеопамять 

				int total_cols = n - (k + 1); //сколько столбцов надо обработать
				if (total_cols > 0)
				{
					int blockSize = 256; // Размер блока потоков
					int gridSize = (total_cols + blockSize - 1) / blockSize; //количество блоков

					// Вызываем наше Кэхэн-ядро!
					hipLaunchKernelGGL(KahanWorker<T>, dim3(gridSize), dim3(blockSize), 0, 0, d_A, d_v, a, n, k); //вызываем воркер для ядра GPU

					hipDeviceSynchronize(); //процессор ждет видеокарту
				}

				if (k + 1 < n) // СИНХРОНИЗАЦИЯ ПАМЯТИ !!! GPU -> CPU, CPU считает вектор v, и с его помощью обновляет b, и k-й столбец. CPU должен знать только k+1 столбец, поэтому только его и копируем
				{
					int count = n - (k + 1);
					hipMemcpy2D(&A[(k + 1) * n + (k + 1)], n * sizeof(T), &d_A[(k + 1) * n + (k + 1)], n * sizeof(T), sizeof(T), count, hipMemcpyDeviceToHost); //эффективное копирование, программа умирает если копировать в цикле
				}

				hipFree(d_v); //освобождаем память под вектор v



				A[k * n + k] = -sign * norm_x; // мы всегда знаем каким будет "первый" столбец
				for (int i = k + 1; i < n; i++) 
				{ 
					A[i * n + k] = T(0);
				}
				int count_send = n - k;
				hipMemcpy2D(&d_A[k * n + k], n * sizeof(T), &A[k * n + k], n * sizeof(T), sizeof(T), count_send, hipMemcpyHostToDevice); //СИНХРОНИЗИРОВАТЬ ПАМЯТЬ!!! CPU -> GPU. GPU обновит у себя все столбцы кроме k-го, поэтому его надо скопировать, Столбец лежит в памяти друг за другом!!! - Экономия оперций копирования
			}
			delete[] v;
		}
	}

	// Забираем готовую треугольную матрицу из видеопамяти
	hipMemcpy(A, d_A, n * n * sizeof(T), hipMemcpyDeviceToHost); 	// Забираем готовую треугольную матрицу из видеопамяти
	hipFree(d_A); //освобождаем видеопамять от матрицы A

	T sum = T(0);
	err = T(0);
	for (int i = n - 1; i >= 0; i--) //обратный ход на CPU
	{
		sum = T(0);
		for (int j = i + 1; j < n; j++)
		{
			KahanStepCPU(A[i * n + j] * x[j], sum, err);
		}
		err = T(0);

		if (std::abs(A[i * n + i]) > 1e-20)
		{
			x[i] = (b[i] - sum) / A[i * n + i]; //здесь слишком большая погрешность, не спасает; даже Kahan
		}
		else
		{
			std::cout << "The system is degenerate; there is no unique solution." << '\n';
			return 1;
		}
	}

	return 0;
}


template <typename T>
__global__ void VectorWorkerGPUONLY(T* d_A, T* d_b, T* d_v, T* d_a, int n, int k) //воркер для подсчета вектора v и коэффициента a
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		T norm_x = T(0);
		T err = T(0);

		for (int i = k; i < n; i++)
		{
			KahanStepGPU(d_A[i * n + k] * d_A[i * n + k], norm_x, err);
		}
		norm_x = std::sqrt(norm_x);

		if (norm_x > T(0))
		{
			T sign = (d_A[k * n + k] >= T(0)) ? T(1) : T(-1);
			d_v[0] = d_A[k * n + k] + sign * norm_x;

			T norm_v_2 = d_v[0] * d_v[0];
			T err_v = T(0);
			for (int i = k + 1; i < n; i++)
			{
				d_v[i - k] = d_A[i * n + k];
				KahanStepGPU(d_v[i - k] * d_v[i - k], norm_v_2, err_v);
			}

			if (norm_v_2 > T(0))
			{
				*d_a = T(2) / norm_v_2;

				T product_b = d_v[0] * d_b[k];
				T err_b = T(0);
				for (int i = k + 1; i < n; i++)
				{
					KahanStepGPU(d_v[i - k] * d_b[i], product_b, err_b);
				}

				d_b[k] -= (*d_a) * d_v[0] * product_b;
				for (int i = k + 1; i < n; i++)
				{
					d_b[i] -= (*d_a) * d_v[i - k] * product_b;
				}
			}
			else
			{
				*d_a = T(0);
			}

			d_A[k * n + k] = -sign * norm_x;
			for (int i = k + 1; i < n; i++)
			{
				d_A[i * n + k] = T(0);
			}
		}
		else
		{
			*d_a = T(0);
		}
	}
}


template <typename T>
__global__ void KahanWorkerGPUONLY(T* d_A, const T* d_v, const T* d_a, int n, int k) //воркер для потока карты, каждый поток обрабатывает свой столбец
{
	int j = blockIdx.x * blockDim.x + threadIdx.x + (k + 1);

	if (j < n)
	{
		T a = *d_a; //коэффициент a
		if (a > T(0))
		{
			T product_A = T(0);
			T err_prod = T(0);

			KahanStepGPU(d_v[0] * d_A[k * n + j], product_A, err_prod);
			for (int i = k + 1; i < n; i++)
			{
				KahanStepGPU(d_v[i - k] * d_A[i * n + j], product_A, err_prod);
			}

			d_A[k * n + j] -= d_v[0] * a * product_A;
			for (int i = k + 1; i < n; i++)
			{
				d_A[i * n + j] -= a * d_v[i - k] * product_A;
			}
		}
	}
}

template <typename T>
int SolveHIPKahanGPUONLY(T* A, const int n, T* b, T* x)
{
	hipSetDevice(0);

	T* d_A = nullptr; //матрицв A
	T* d_b = nullptr; //вектор b
	T* d_v = nullptr; //вектор v
	T* d_a = nullptr; //коэффициент a

	hipMalloc(&d_A, n * n * sizeof(T)); //выделяем видеопаямть 
	hipMalloc(&d_b, n * sizeof(T));
	hipMalloc(&d_v, n * sizeof(T));
	hipMalloc(&d_a, sizeof(T));

	hipMemcpy(d_A, A, n * n * sizeof(T), hipMemcpyHostToDevice); //копируем A и b 
	hipMemcpy(d_b, b, n * sizeof(T), hipMemcpyHostToDevice);

	for (int k = 0; k < n - 1; k++)
	{
		hipLaunchKernelGGL(VectorWorkerGPUONLY<T>, dim3(1), dim3(1), 0, 0, d_A, d_b, d_v, d_a, n, k); //считаем вектор v, коэффициент a, обновляем b (карта сама синхронизируется)

		int total_cols = n - (k + 1);
		if (total_cols > 0)
		{
			int blockSize = 256; //размер блока потоков
			int gridSize = (total_cols + blockSize - 1) / blockSize; //кол-во блоков

			hipLaunchKernelGGL(KahanWorkerGPUONLY<T>, dim3(gridSize), dim3(blockSize), 0, 0, d_A, d_v, d_a, n, k); //обновляем A (карта сама синхронизируется)
		}
	}

	hipDeviceSynchronize(); //ждем пока карта все досчитает

	hipMemcpy(A, d_A, n * n * sizeof(T), hipMemcpyDeviceToHost); //копируем GPU -> CPU
	hipMemcpy(b, d_b, n * sizeof(T), hipMemcpyDeviceToHost);

	hipFree(d_A); //освобождаем видеопамять
	hipFree(d_b);
	hipFree(d_v);
	hipFree(d_a);

	T sum = T(0);
	T err = T(0);
	for (int i = n - 1; i >= 0; i--) //обратный ход на CPU
	{
		sum = T(0);
		for (int j = i + 1; j < n; j++)
		{
			KahanStepCPU(A[i * n + j] * x[j], sum, err);
		}
		err = T(0);

		if (std::abs(A[i * n + i]) > 1e-20)
		{
			x[i] = (b[i] - sum) / A[i * n + i];
		}
		else
		{
			std::cout << "The system is degenerate; there is no unique solution." << '\n';
			return 1;
		}
	}

	return 0;
}