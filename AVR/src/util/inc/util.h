#ifndef _UTIL_UTIL_H_
#define _UTIL_UTIL_H_

#include <stddef.h>
#include <inttypes.h>

#include <avr/io.h>
#include <avr/interrupt.h>

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

extern void sleep(uint64_t ms);

#ifdef UTIL_DEFINE_SLEEP
#include <util/delay.h>
void sleep(uint64_t ms) { ms /= 50; while (ms--) _delay_ms(1); }
#endif

extern void* operator new  (size_t size);
extern void* operator new[](size_t size);
extern void  operator delete  (void* ptr);
extern void  operator delete[](void* ptr);
extern void  operator delete  (void* ptr, size_t size);
extern void  operator delete[](void* ptr, size_t size);

#endif
