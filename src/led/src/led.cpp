#include "led.h"

Led::Led(Bit led_port, bool inverted)
	: m_port(led_port), m_inverted(inverted)
{
	Pin(led_port).dd.set();
}

Led::~Led()
{
}

bool Led::operator=(bool state)
{
	if (state ^ m_inverted)
		m_port.set();
	else
		m_port.clear();

	return state;
}
