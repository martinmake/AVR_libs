#ifndef _LED_LED_H_
#define _LED_LED_H_

#include <gpio.h>

enum class POLARITY : bool
{
	NONINVERTED, INVERTED
};

template <PORT port, uint8_t index, POLARITY polarity = POLARITY::NONINVERTED>
class Led : protected Gpio<port, index>
{
	public: // CONSTRUCTORS
		Led(void);

	public: // GETTERS
		bool is_on (void);
		bool is_off(void);

	public: // FUNCTIONS
		void toggle  (void);
		void turn_on (void);
		void turn_off(void);

	public: // OPERATORS
		Led& operator=(bool state);
		operator bool(void);
};

template <PORT port, uint8_t index, POLARITY polarity>
Led<port, index, polarity>::Led(void)
	: Gpio<port, index>(MODE::OUTPUT)
{
	turn_off();
}

template <PORT port, uint8_t index, POLARITY polarity>
Led<port, index, polarity>& Led<port, index, polarity>::operator=(bool state)
{
	switch (state)
	{
		case ON:  turn_on (); break;
		case OFF: turn_off(); break;
	}

	return *this;
}
template <PORT port, uint8_t index, POLARITY polarity>
Led<port, index, polarity>::operator bool(void)
{
	return is_on() ? ON : OFF;
}

template <PORT port, uint8_t index, POLARITY polarity>
void Led<port, index, polarity>::turn_on(void)
{
	switch (polarity)
	{
		case POLARITY::NONINVERTED: this->set  (); break;
		case POLARITY::INVERTED:    this->clear(); break;
	}
}
template <PORT port, uint8_t index, POLARITY polarity>
void Led<port, index, polarity>::turn_off(void)
{
	switch (polarity)
	{
		case POLARITY::NONINVERTED: this->clear(); break;
		case POLARITY::INVERTED:    this->set  (); break;
	}
}
template <PORT port, uint8_t index, POLARITY polarity>
void Led<port, index, polarity>::toggle(void)
{
	if (is_on())
		turn_off();
	else
		turn_on();
}

template <PORT port, uint8_t index, POLARITY polarity>
bool Led<port, index, polarity>::is_on(void)
{
	switch (polarity)
	{
		case POLARITY::NONINVERTED: return this->is_high();
		case POLARITY::INVERTED:    return this->is_low ();
	}
}

#endif
