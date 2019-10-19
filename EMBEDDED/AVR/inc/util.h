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

#define OK  true
#define ERR false

#define HIGH true
#define LOW  false

#define ACK  true
#define NACK false

inline bool __sei(void) { sei(); return true;  }
inline bool __cli(void) { cli(); return false; }

#define ATOMIC() for (bool run = __cli(); run; run = __sei())

extern void* operator new  (size_t size);
extern void* operator new[](size_t size);
extern void  operator delete  (void* ptr);
extern void  operator delete[](void* ptr);
extern void  operator delete  (void* ptr, size_t size);
extern void  operator delete[](void* ptr, size_t size);

#endif
