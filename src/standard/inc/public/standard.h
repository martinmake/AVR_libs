#ifndef _STANDARD_STANDARD_H_
#define _STANDARD_STANDARD_H_

#include <stddef.h>
#include <inttypes.h>

#include "bit.h"
#include "pin.h"

typedef enum {
	BIN =  2,
	OCT =  8,
	DEC = 10,
	HEX = 16
} BASE;

extern void* operator new(size_t size);
extern void* operator new[](size_t size);
extern void  operator delete(void* ptr);
extern void  operator delete(void* ptr, size_t sz);
extern void  operator delete[](void* ptr);
extern void  operator delete[](void* ptr, size_t sz);

#endif
