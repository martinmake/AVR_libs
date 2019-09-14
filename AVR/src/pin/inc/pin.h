#ifndef _PIN_PIN_H_
#define _PIN_PIN_H_

#include <inttypes.h>
#include <avr/io.h>

#include <util.h>

enum class PORT : uint8_t
{
	B, C, D
};
enum class DIRECTION : uint8_t
{
	INPUT, OUTPUT
};

template <PORT port, uint8_t index, DIRECTION direction = DIRECTION::OUTPUT>
class Pin
{
	public: // CONSTRUCTORS
		Pin(void);

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
		Pin<port, index, direction>& operator=(STATE state);
		operator STATE(void);
		operator bool (void);
};

template <PORT port, uint8_t index, DIRECTION direction>
Pin<port, index, direction>::Pin(void)
{
	switch (direction)
	{
		case DIRECTION::OUTPUT: make_output(); break;
		case DIRECTION:: INPUT: make_input (); break;
	}
}

template <PORT port, uint8_t index, DIRECTION direction>
Pin<port, index, direction>& Pin<port, index, direction>::operator=(STATE state)
{
	switch (state)
	{
		case HIGH: set  (); break;
		case LOW:  clear(); break;
	}

	return *this;
}
template <PORT port, uint8_t index, DIRECTION direction>
Pin<port, index, direction>::operator STATE(void)
{
	return is_high() ? HIGH : LOW;
}
template <PORT port, uint8_t index, DIRECTION direction>
Pin<port, index, direction>::operator bool(void)
{
	return is_high();
}

template <PORT port, uint8_t index, DIRECTION direction>
void Pin<port, index, direction>::set(void)
{
	switch (port)
	{
		case PORT::B: PORTB |= (1 << index); break;
		case PORT::C: PORTC |= (1 << index); break;
		case PORT::D: PORTD |= (1 << index); break;
	}
}
template <PORT port, uint8_t index, DIRECTION direction>
void Pin<port, index, direction>::clear(void)
{
	switch (port)
	{
		case PORT::B: PORTB &= ~(1 << index); break;
		case PORT::C: PORTC &= ~(1 << index); break;
		case PORT::D: PORTD &= ~(1 << index); break;
	}
}

template <PORT port, uint8_t index, DIRECTION direction>
void Pin<port, index, direction>::make_output(void)
{
	switch (port)
	{
		case PORT::B: DDRB |= (1 << index); break;
		case PORT::C: DDRC |= (1 << index); break;
		case PORT::D: DDRD |= (1 << index); break;
	}
}
template <PORT port, uint8_t index, DIRECTION direction>
void Pin<port, index, direction>::make_input(void)
{
	switch (port)
	{
		case PORT::B: DDRB &= ~(1 << index); break;
		case PORT::C: DDRC &= ~(1 << index); break;
		case PORT::D: DDRD &= ~(1 << index); break;
	}
}

template <PORT port, uint8_t index, DIRECTION direction>
bool Pin<port, index, direction>::is_high(void)
{
	switch (port)
	{
		case PORT::B: return PINB & (1 << index);
		case PORT::C: return PINC & (1 << index);
		case PORT::D: return PIND & (1 << index);
	}
}
template <PORT port, uint8_t index, DIRECTION direction>
bool Pin<port, index, direction>::is_low(void)
{
	return !is_high();
}

template <PORT port, uint8_t index, DIRECTION direction>
bool Pin<port, index, direction>::is_output(void)
{
	switch (port)
	{
		case PORT::B: return DDRB & (1 << index);
		case PORT::C: return DDRC & (1 << index);
		case PORT::D: return DDRD & (1 << index);
	}
}
template <PORT port, uint8_t index, DIRECTION direction>
bool Pin<port, index, direction>::is_input(void)
{
	return !is_output();
}

#endif
