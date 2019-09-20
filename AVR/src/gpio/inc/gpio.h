#ifndef _GPIO_GPIO_H_
#define _GPIO_GPIO_H_

#include <inttypes.h>
#include <avr/io.h>

#include <util.h>

enum class PORT      : uint8_t { B, C, D };
enum class DIRECTION : uint8_t { INPUT, OUTPUT };

template <PORT port, uint8_t index, DIRECTION direction = DIRECTION::OUTPUT>
class Gpio
{
	public: // CONSTRUCTORS
		Gpio(void);

	public: // FUNCTIONS
		void set  (void);
		void clear(void);
		bool is_high(void);
		bool is_low (void);

		void make_input (void);
		void make_output(void);
		bool is_input (void);
		bool is_output(void);

	public: // OPERATORS
		Gpio<port, index, direction>& operator=(STATE state);
		operator STATE(void);
		operator bool (void);
};

template <PORT port, uint8_t index, DIRECTION direction>
Gpio<port, index, direction>::Gpio(void)
{
	switch (direction)
	{
		case DIRECTION::OUTPUT: make_output(); break;
		case DIRECTION:: INPUT: make_input (); break;
	}
}

template <PORT port, uint8_t index, DIRECTION direction>
Gpio<port, index, direction>& Gpio<port, index, direction>::operator=(STATE state)
{
	switch (state)
	{
		case HIGH: set  (); break;
		case LOW:  clear(); break;
	}

	return *this;
}
template <PORT port, uint8_t index, DIRECTION direction>
Gpio<port, index, direction>::operator STATE(void)
{
	return is_high() ? HIGH : LOW;
}
template <PORT port, uint8_t index, DIRECTION direction>
Gpio<port, index, direction>::operator bool(void)
{
	return is_high();
}

template <PORT port, uint8_t index, DIRECTION direction>
void Gpio<port, index, direction>::set(void)
{
	switch (port)
	{
		case PORT::B: PORTB |= (1 << index); break;
		case PORT::C: PORTC |= (1 << index); break;
		case PORT::D: PORTD |= (1 << index); break;
	}
}
template <PORT port, uint8_t index, DIRECTION direction>
void Gpio<port, index, direction>::clear(void)
{
	switch (port)
	{
		case PORT::B: PORTB &= ~(1 << index); break;
		case PORT::C: PORTC &= ~(1 << index); break;
		case PORT::D: PORTD &= ~(1 << index); break;
	}
}

template <PORT port, uint8_t index, DIRECTION direction>
void Gpio<port, index, direction>::make_output(void)
{
	switch (port)
	{
		case PORT::B: DDRB |= (1 << index); break;
		case PORT::C: DDRC |= (1 << index); break;
		case PORT::D: DDRD |= (1 << index); break;
	}
}
template <PORT port, uint8_t index, DIRECTION direction>
void Gpio<port, index, direction>::make_input(void)
{
	switch (port)
	{
		case PORT::B: DDRB &= ~(1 << index); break;
		case PORT::C: DDRC &= ~(1 << index); break;
		case PORT::D: DDRD &= ~(1 << index); break;
	}
}

template <PORT port, uint8_t index, DIRECTION direction>
bool Gpio<port, index, direction>::is_high(void)
{
	switch (port)
	{
		case PORT::B: return PORTB & (1 << index);
		case PORT::C: return PORTC & (1 << index);
		case PORT::D: return PORTD & (1 << index);
	}
}
template <PORT port, uint8_t index, DIRECTION direction>
bool Gpio<port, index, direction>::is_low(void)
{
	return !is_high();
}

template <PORT port, uint8_t index, DIRECTION direction>
bool Gpio<port, index, direction>::is_output(void)
{
	switch (port)
	{
		case PORT::B: return DDRB & (1 << index);
		case PORT::C: return DDRC & (1 << index);
		case PORT::D: return DDRD & (1 << index);
	}
}
template <PORT port, uint8_t index, DIRECTION direction>
bool Gpio<port, index, direction>::is_input(void)
{
	return !is_output();
}

#endif
