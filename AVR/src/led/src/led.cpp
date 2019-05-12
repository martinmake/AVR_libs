#include "led.h"

Led::Led(Pin& led_pin, bool inverted)
	: m_port(Bit(led_pin.port)), m_inverted(inverted)
{
	led_pin.dd.set();
}

Led::Led(Pin led_pin, bool inverted)
	: m_port(Bit(led_pin.port)), m_inverted(inverted)
{
	led_pin.dd.set();
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
