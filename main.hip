#include "SolveHIP.h"
#include "initHIP.h"

#include<iostream>
#include<fstream>
#include<algorithm>
#include<iomanip>
#include<cmath>
#include<string>
#include<chrono>



int main(int argc, char* argv[])
{
	int k, n, m; // формула матрицы, размерность матрицы, кол-во выводимых значений

	if (argc < 4)
	{
		std::cout << "There are not enough arguments" << '\n';
		return 1;
	}

	try
	{
		n = std::stoi(argv[1]);
		m = std::stoi(argv[2]);
		k = std::stoi(argv[3]);
	}
	catch (...)
	{
		std::cout << "Invalid parameter format" << '\n';
		return 1;
	}

	if ((n <= 0) || (m <= 0) || (k < 0) || (k > 4))
	{
		std::cout << "Invalid parameter format" << '\n';
		return 1;
	}

	std::string filename = "";

	if (k == 0)
	{
		if (argc < 5)
		{
			std::cout << "There are not enough arguments" << '\n';
			return 1;
		}
		filename = argv[4];
	}
	else
	{
		if (argc >= 5)
		{
			std::cout << "An excess of arguments" << '\n';
			return 1;
		}
	}

	std::cout << "INPUT PARAMETERS:" << '\n';
	std::cout << "Matrix size (N): " << n << '\n';
	std::cout << "Output count (M): " << m << '\n';
	std::cout << "Formula type (K): " << k << '\n';

	if (hipSetDevice(0) != hipSuccess)
	{
		std::cout << "Error: AMD Radeon not found!" << '\n';
		return 1;
	}

	std::cout << "Size of double: " << sizeof(double) << " bytes" << std::endl;
	std::cout << "Size of long double: " << sizeof(long double) << " bytes" << std::endl;

	std::cout << '\n' << "type: double + GPU" << '\n' << std::endl;

	{
		double* A = new double[n * n]; // Матрица
		double* b = new double[n]; // Правая часть
		double* x = new double[n]; // Решение
		int q;

		auto start_time = std::chrono::high_resolution_clock::now();
		q = Init<double>(A, b, k, n, m, filename);
		auto end_time = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> duration = end_time - start_time;
		std::cout << std::defaultfloat;
		std::cout << "System initialization time: " << (duration.count()) * 100.0 << " hundredths of a second" << '\n';

		if (q == 0)
		{
			std::cout << "Matrix: " << '\n';
			printMatrix<double>(A, n, n, m);


			start_time = std::chrono::high_resolution_clock::now();
			q = SolveHIPKahanGPUONLY<double>(A, n, b, x);
			end_time = std::chrono::high_resolution_clock::now();

			if (q != 1)
			{
				duration = end_time - start_time;
				std::cout << std::defaultfloat;
				std::cout << "System solution time: " << (duration.count()) * 100.0 << " hundredths of a second" << '\n';

				std::cout << "Solution: " << '\n';
				printMatrix<double>(x, n, 1, m);

				q = Init<double>(A, b, k, n, m, filename);
				if (q == 0)
				{
					std::cout << "Norm of residual: " << std::scientific << std::setprecision(3) << CalcNormNevKahan<double>(A, b, x, n) << '\n';

					std::cout << "Solution error margin: " << CalcErNormKahan<double>(x, n) << '\n';
				}
			}
		}
		delete[] A;
		delete[] b;
		delete[] x;
	}
	return 0;
}