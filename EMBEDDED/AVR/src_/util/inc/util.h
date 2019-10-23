#ifndef _UTIL_UTIL_H_
#define _UTIL_UTIL_H_

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <avr/io.h>
#include <avr/interrupt.h>
#ifdef F_CPU
#include <util/delay.h>
#endif
#include <util/atomic.h>
#include <avr/pgmspace.h>

#include <math/util.h>

#include "util/tty.h"

#define BIT(index) (1 << index)

#define SET(   port, index) port |=  BIT(index)
#define CLEAR( port, index) port &= ~BIT(index)
#define TOOGLE(port, index) port ^=  BIT(index)

#define IS_SET(  port, index) (  port & BIT(index) )
#define IS_CLEAR(port, index) (!(port & BIT(index)))

#define BIN =  2,
#define OCT =  8,
#define DEC = 10,
#define HEX = 16,

#define ON  true
#define OFF false

#define ENABLE  true
#define DISABLE false

#define OK  true
#define ERR false

#define HIGH true
#define LOW  false

#define ACK  true
#define NACK false

inline bool __sei(void) { sei(); return true;  }
inline bool __cli(void) { cli(); return false; }

#define ATOMIC() for (bool run = __cli(); run; run = __sei())

extern void assert_failed(const char* exp, const char* file, int line);
#define assert(exp) if (!(exp)) assert_failed(#exp, __FILE__, __LINE__);
#ifndef CUSTOM_ASSERT
inline void assert_failed(const char* exp, const char* file, int line)
{
	tty_escape_sequence(FG_RED, BOLD);
	fprintf(stderr, "%s:%d: ASSERTION `%s' FAILED\n", file, line, exp);
	tty_escape_sequence(NORMAL);
}
#endif

extern void* operator new  (size_t size);
extern void* operator new[](size_t size);
extern void  operator delete  (void* ptr);
extern void  operator delete[](void* ptr);
extern void  operator delete  (void* ptr, size_t size);
extern void  operator delete[](void* ptr, size_t size);

#endif
