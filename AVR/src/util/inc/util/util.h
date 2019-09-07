#ifndef _UTIL_UTIL_H_
#define _UTIL_UTIL_H_

#include <stddef.h>
#include <inttypes.h>

typedef enum {
	BIN =  2,
	OCT =  8,
	DEC = 10,
	HEX = 16
} BASE;

#define BIT(index) (1 << index)

extern void sleep(uint64_t ms);

#ifdef UTIL_DEFINE_SLEEP
#include <util/delay.h>
void sleep(uint64_t ms) { while (ms--) _delay_ms(1); }
#endif

extern void* operator new  (size_t size);
extern void* operator new[](size_t size);
extern void  operator delete  (void* ptr);
extern void  operator delete[](void* ptr);
extern void  operator delete  (void* ptr, size_t size);
extern void  operator delete[](void* ptr, size_t size);

#endif
