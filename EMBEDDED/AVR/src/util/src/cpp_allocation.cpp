#include <stdlib.h>

extern "C" void __cxa_pure_virtual() { while (1); }

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

void operator delete(void* ptr, size_t size)
{
	(void) size;
	free(ptr);
}

void operator delete[](void* ptr)
{
	free(ptr);
}

void operator delete[](void* ptr, size_t size)
{
	(void) size;
	free(ptr);
}
