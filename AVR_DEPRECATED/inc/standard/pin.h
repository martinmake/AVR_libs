#ifndef _STANDARD_PIN_H_
#define _STANDARD_PIN_H_

#include "bit.h"

typedef enum direction {
	INPUT  = 0,
	OUTPUT = 1
} DIRECTION;

typedef enum state {
	LOW  = 0,
	HIGH = 1
} STATE;


class Pin
{
	public:
		Bit port;
		Bit dd;
		Bit pin;

	public:
		Pin(const Bit& port_bit, DIRECTION direction = INPUT);
		~Pin();

		inline Pin& operator=(uint8_t val) { port = val; return *this; }
};

#endif
