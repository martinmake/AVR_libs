#ifndef _UTIL_UTIL_H_
#define _UTIL_UTIL_H_

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#include <avr/io.h>
#include <avr/interrupt.h>
#ifdef F_CPU
#include <util/delay.h>
#endif
#include <util/atomic.h>

#include <math/util.h>

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

#define ON   true
#define OFF  false

#define HIGH true
#define LOW  false

#define ACK  true
#define NACK false

extern void* operator new  (size_t size);
extern void* operator new[](size_t size);
extern void  operator delete  (void* ptr);
extern void  operator delete[](void* ptr);
extern void  operator delete  (void* ptr, size_t size);
extern void  operator delete[](void* ptr, size_t size);

#endif
