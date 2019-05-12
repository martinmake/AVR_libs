#include <stdlib.h>

#include "standard.h"

void* operator new(size_t size)
{
	return malloc(size);
}

void* operator new[](size_t size)
{
	return malloc(size);
}

void operator delete(void* ptr)
{
	free(ptr);
}

void operator delete(void* ptr, size_t sz)
{
	(void) sz;
	free(ptr);
}

void operator delete[](void* ptr)
{
	free(ptr);
}

void operator delete[](void* ptr, size_t sz)
{
	(void) sz;
	free(ptr);
}
