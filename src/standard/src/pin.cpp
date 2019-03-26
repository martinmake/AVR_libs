#ifndef _STANDARD_PIN_H_
#define _STANDARD_PIN_H_

#include "standard.h"

Pin::Pin(const Bit& port_reg)
	: port(port_reg), dd(port_reg-1), pin(port_reg-2)
{
}

Pin::~Pin()
{
}

#endif
