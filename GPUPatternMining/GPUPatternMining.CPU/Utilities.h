#pragma once

class Utilities
{
public:
	static unsigned long long int factorial(unsigned long long int n)
	{
		return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
	}
};