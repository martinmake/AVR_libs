#ifndef _GPIO_GPIO_H_
#define _GPIO_GPIO_H_

#include <inttypes.h>
#include <avr/io.h>

#include <util.h>

enum class PORT : uint8_t { B, C, D };
enum       MODE : uint8_t { INPUT, OUTPUT };

template <PORT port, uint8_t index>
class Gpio
{
	public: // CONSTRUCTORS
		Gpio(MODE mode = MODE::OUTPUT);

	public: // FUNCTIONS
		void set  (void);
		void clear(void);
		bool is_high(void);
		bool is_low (void);

		void make_input (void);
		void make_output(void);
		bool is_input (void);
		bool is_output(void);

		void pull_up   (void);
		void disconnect(void);
		bool is_pulled_up   (void);
		bool is_disconnected(void);

	public: // OPERATORS
		Gpio<port, index>& operator=(bool state);
		operator bool(void);
};

template <PORT port, uint8_t index>
Gpio<port, index>::Gpio(MODE mode)
{
	switch (mode)
	{
		case MODE::OUTPUT: make_output(); break;
		case MODE:: INPUT: make_input (); break;
	}
}

template <PORT port, uint8_t index>
Gpio<port, index>& Gpio<port, index>::operator=(bool state)
{
	switch (state)
	{
		case HIGH: set  (); break;
		case LOW:  clear(); break;
	}

	return *this;
}
template <PORT port, uint8_t index>
Gpio<port, index>::operator bool(void)
{
	return is_high();
}

template <PORT port, uint8_t index>
void Gpio<port, index>::set(void)
{
	switch (port)
	{
		case PORT::B: SET(PORTB, index); break;
		case PORT::C: SET(PORTC, index); break;
		case PORT::D: SET(PORTD, index); break;
	}
}
template <PORT port, uint8_t index>
void Gpio<port, index>::clear(void)
{
	switch (port)
	{
		case PORT::B: CLEAR(PORTB, index); break;
		case PORT::C: CLEAR(PORTC, index); break;
		case PORT::D: CLEAR(PORTD, index); break;
	}
}

template <PORT port, uint8_t index>
void Gpio<port, index>::make_output(void)
{
	switch (port)
	{
		case PORT::B: SET(DDRB, index); break;
		case PORT::C: SET(DDRC, index); break;
		case PORT::D: SET(DDRD, index); break;
	}
}
template <PORT port, uint8_t index>
void Gpio<port, index>::make_input(void)
{
	switch (port)
	{
		case PORT::B: CLEAR(DDRB, index); break;
		case PORT::C: CLEAR(DDRC, index); break;
		case PORT::D: CLEAR(DDRD, index); break;
	}
}

template <PORT port, uint8_t index>
bool Gpio<port, index>::is_high(void)
{
	switch (port)
	{
		case PORT::B: return IS_SET(PINB, index);
		case PORT::C: return IS_SET(PINC, index);
		case PORT::D: return IS_SET(PIND, index);
	}
}
template <PORT port, uint8_t index>
bool Gpio<port, index>::is_low(void)
{
	switch (port)
	{
		case PORT::B: return IS_CLEAR(PINB, index);
		case PORT::C: return IS_CLEAR(PINC, index);
		case PORT::D: return IS_CLEAR(PIND, index);
	}
}

template <PORT port, uint8_t index>
bool Gpio<port, index>::is_output(void)
{
	switch (port)
	{
		case PORT::B: return IS_SET(DDRB, index);
		case PORT::C: return IS_SET(DDRC, index);
		case PORT::D: return IS_SET(DDRD, index);
	}
}
template <PORT port, uint8_t index>
bool Gpio<port, index>::is_input(void)
{
	switch (port)
	{
		case PORT::B: return IS_CLEAR(DDRB, index);
		case PORT::C: return IS_CLEAR(DDRC, index);
		case PORT::D: return IS_CLEAR(DDRD, index);
	}
}

template <PORT port, uint8_t index>
void Gpio<port, index>::pull_up(void)
{
	switch (port)
	{
		case PORT::B: SET(PORTB, index);
		case PORT::C: SET(PORTC, index);
		case PORT::D: SET(PORTD, index);
	}
}
template <PORT port, uint8_t index>
void Gpio<port, index>::disconnect(void)
{
	switch (port)
	{
		case PORT::B: CLEAR(PORTB, index);
		case PORT::C: CLEAR(PORTC, index);
		case PORT::D: CLEAR(PORTD, index);
	}
}
template <PORT port, uint8_t index>
bool Gpio<port, index>::is_pulled_up(void)
{
	switch (port)
	{
		case PORT::B: return IS_SET(PORTB, index);
		case PORT::C: return IS_SET(PORTC, index);
		case PORT::D: return IS_SET(PORTD, index);
	}
}
template <PORT port, uint8_t index>
bool Gpio<port, index>::is_disconnected(void)
{
	switch (port)
	{
		case PORT::B: return IS_CLEAR(PORTB, index);
		case PORT::C: return IS_CLEAR(PORTC, index);
		case PORT::D: return IS_CLEAR(PORTD, index);
	}
}

#endif
