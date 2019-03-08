#ifndef _STANDARD_STANDARD_H_
#define _STANDARD_STANDARD_H_

#include <stddef.h>
#include <inttypes.h>

#ifdef __cplusplus
#define CAST
#else
#define CAST (BIT)
#endif
#define PRT(bit) CAST {bit.addr  , bit.index}
#define DDR(bit) CAST {bit.addr-1, bit.index}
#define PIN(bit) CAST {bit.addr-2, bit.index}

typedef enum {
	BIN =  2,
	OCT =  8,
	DEC = 10,
	HEX = 16
} BASE;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	volatile uint8_t* addr;
	uint8_t  index;
} BIT;

extern void set_bit(BIT bit);
extern void clear_bit(BIT bit);
extern void write_bit(BIT bit, uint8_t val);
extern uint8_t read_bit(BIT bit);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern void* operator new(size_t size);
extern void* operator new[](size_t size);
extern void  operator delete(void* ptr);
extern void  operator delete[](void* ptr);
#endif

#endif
