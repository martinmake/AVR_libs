#ifndef _STANDARD_PIN_H_
#define _STANDARD_PIN_H_

#include "standard.h"

Pin::Pin(const Bit& port)
	: port(port), dd(port-1), pin(port-2)
{
}

Pin::~Pin()
{
}

#endif
