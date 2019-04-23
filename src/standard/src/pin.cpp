#include "pin.h"

Pin::Pin(const Bit& port_bit)
	: port(port_bit, -0),
	  dd  (port_bit, -1),
	  pin (port_bit, -2)
{
}

Pin::Pin(volatile uint8_t& reg, uint8_t index)
	: port(*(&reg - 0), index),
	  dd  (*(&reg - 1), index),
	  pin (*(&reg - 2), index)
{
}

Pin::~Pin()
{
}
