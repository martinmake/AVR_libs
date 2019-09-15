#ifndef _UTIL_UTIL_H_
#define _UTIL_UTIL_H_

#include <stddef.h>
#include <inttypes.h>

#include <avr/io.h>
#include <avr/interrupt.h>
#ifdef F_CPU
#include <util/delay.h>
#endif

#define BIT(index) (1 << index)

#define SET(  port, index) port |=  BIT(index)
#define CLEAR(port, index) port &= ~BIT(index)

#define IS_SET(  port, index) (  port & BIT(index) )
#define IS_CLEAR(port, index) (!(port & BIT(index)))

typedef enum {
	BIN =  2,
	OCT =  8,
	DEC = 10,
	HEX = 16,
} BASE;
typedef enum {
	OFF = 0, ON   = 1,
	LOW = 0, HIGH = 1,
} STATE;

extern void* operator new  (size_t size);
extern void* operator new[](size_t size);
extern void  operator delete  (void* ptr);
extern void  operator delete[](void* ptr);
extern void  operator delete  (void* ptr, size_t size);
extern void  operator delete[](void* ptr, size_t size);

#endif
