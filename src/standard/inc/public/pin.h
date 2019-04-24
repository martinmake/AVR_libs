#ifndef _STANDARD_PIN_H_
#define _STANDARD_PIN_H_

#include "bit.h"

#define INPUT  ((uint8_t) 0)
#define OUTPUT ((uint8_t) 1)

#define LOW  ((uint8_t) 0)
#define HIGH ((uint8_t) 1)

class Pin
{
	public:
		Bit port;
		Bit dd;
		Bit pin;

	public:
		Pin(const Bit& port_bit);
		Pin(volatile uint8_t& reg, uint8_t index);
		~Pin();
};

#endif
