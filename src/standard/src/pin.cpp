#ifndef _STANDARD_PIN_H_
#define _STANDARD_PIN_H_

#include "standard.h"

Pin::Pin(volatile uint8_t& port_reg, uint8_t index)
	: port(port_reg, index), dd(*(&port_reg-1), index), pin(*(&port_reg-2), index)
{
}

Pin::~Pin()
{
}

#endif
