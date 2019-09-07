#ifndef _LED_LED_H_
#define _LED_LED_H_

#include <pin/pin.h>

enum class Polarity : bool
{
	NONINVERTED, INVERTED
};

template <Port port, uint8_t index, Polarity polarity = Polarity::NONINVERTED>
class Led : protected Pin<port, index, Direction::OUTPUT>
{
	public: // CONSTRUCTORS
		Led(void);

	public: // FUNCTIONS
		void turn_on (void);
		void turn_off(void);
		bool is_on (void);
		bool is_off(void);

	public: // OPERATORS
		Led& operator=(STATE state);
		operator STATE(void);
		operator bool(void);
};

template <Port port, uint8_t index, Polarity polarity>
Led<port, index, polarity>::Led(void)
	: Pin<port, index, Direction::OUTPUT>()
{
}

template <Port port, uint8_t index, Polarity polarity>
Led<port, index, polarity>& Led<port, index, polarity>::operator=(STATE state)
{
	switch (state)
	{
		case ON:  turn_on (); break;
		case OFF: turn_off(); break;
	}

	return *this;
}
template <Port port, uint8_t index, Polarity polarity>
Led<port, index, polarity>::operator STATE(void)
{
	return is_on() ? ON : OFF;
}

template <Port port, uint8_t index, Polarity polarity>
void Led<port, index, polarity>::turn_on(void)
{
	switch (polarity)
	{
		case Polarity::NONINVERTED: this->set  (); break;
		case Polarity::INVERTED:    this->clear(); break;
	}
}
template <Port port, uint8_t index, Polarity polarity>
void Led<port, index, polarity>::turn_off(void)
{
	switch (polarity)
	{
		case Polarity::NONINVERTED: this->clear(); break;
		case Polarity::INVERTED:    this->set  (); break;
	}
}

template <Port port, uint8_t index, Polarity polarity>
bool Led<port, index, polarity>::is_on(void)
{
	switch (polarity)
	{
		case Polarity::NONINVERTED: return this->is_high();
		case Polarity::INVERTED:    return this->is_low ();
	}
}
template <Port port, uint8_t index, Polarity polarity>
bool Led<port, index, polarity>::is_off(void)
{
	return !is_on();
}

#endif
