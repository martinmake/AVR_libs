#include "pin.h"

Pin::Pin(const Bit& port_bit, DIRECTION direction)
	: port(port_bit, -0),
	  dd  (port_bit, -1),
	  pin (port_bit, -2)
{
	dd = direction;
}

Pin::~Pin()
{
}
