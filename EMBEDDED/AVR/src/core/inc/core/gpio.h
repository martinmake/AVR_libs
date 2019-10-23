#ifndef _GPIO_GPIO_H_
#define _GPIO_GPIO_H_

#include <inttypes.h>
#include <avr/io.h>

#include <util.h>

namespace Core
{
	namespace GPIO
	{
		enum class PORT : uint8_t { B, C, D };
		enum class MODE : uint8_t { INPUT, OUTPUT };
	}

	template <GPIO::PORT port, uint8_t index>
	class Gpio
	{
		public: // CONSTRUCTORS
			Gpio(GPIO::MODE mode = GPIO::MODE::OUTPUT);

		public: // METHODS
			void set  (void);
			void clear(void);
			void toggle(void);
			bool is_high(void) const;
			bool is_low (void) const;

			void make_input (void);
			void make_output(void);
			bool is_input (void) const;
			bool is_output(void) const;

			void pull_up   (void);
			void disconnect(void);
			bool is_pulled_up   (void) const;
			bool is_disconnected(void) const;

		public: // OPERATORS
			Gpio<port, index>& operator=(bool state);
			Gpio<port, index>& operator()(GPIO::MODE mode);
			operator bool(void) const;
	};

	template <GPIO::PORT port, uint8_t index>
	Gpio<port, index>::Gpio(GPIO::MODE mode)
	{
		using namespace GPIO;
		switch (mode)
		{
			case MODE::OUTPUT: make_output(); break;
			case MODE:: INPUT: make_input (); break;
		}
	}

	template <GPIO::PORT port, uint8_t index>
	Gpio<port, index>& Gpio<port, index>::operator=(bool state)
	{
		switch (state)
		{
			case HIGH: set  (); break;
			case LOW:  clear(); break;
		}

		return *this;
	}
	template <GPIO::PORT port, uint8_t index>
	Gpio<port, index>& Gpio<port, index>::operator()(GPIO::MODE mode)
	{
		using namespace GPIO;
		switch (mode)
		{
			case MODE::OUTPUT: make_output(); break;
			case MODE:: INPUT: make_input (); break;
		}

		return *this;
	}
	template <GPIO::PORT port, uint8_t index>
	Gpio<port, index>::operator bool(void) const
	{
		return is_high();
	}

	template <GPIO::PORT port, uint8_t index>
	void Gpio<port, index>::set(void)
	{
		using namespace GPIO;
		switch (port)
		{
			case PORT::B: SET(PORTB, index); break;
			case PORT::C: SET(PORTC, index); break;
			case PORT::D: SET(PORTD, index); break;
		}
	}
	template <GPIO::PORT port, uint8_t index>
	void Gpio<port, index>::clear(void)
	{
		using namespace GPIO;
		switch (port)
		{
			case PORT::B: CLEAR(PORTB, index); break;
			case PORT::C: CLEAR(PORTC, index); break;
			case PORT::D: CLEAR(PORTD, index); break;
		}
	}
	template <GPIO::PORT port, uint8_t index>
	void Gpio<port, index>::toggle(void)
	{
		using namespace GPIO;
		switch (port)
		{
			case PORT::B: TOOGLE(PORTB, index); break;
			case PORT::C: TOOGLE(PORTC, index); break;
			case PORT::D: TOOGLE(PORTD, index); break;
		}
	}

	template <GPIO::PORT port, uint8_t index>
	void Gpio<port, index>::make_output(void)
	{
		using namespace GPIO;
		switch (port)
		{
			case PORT::B: SET(DDRB, index); break;
			case PORT::C: SET(DDRC, index); break;
			case PORT::D: SET(DDRD, index); break;
		}
	}
	template <GPIO::PORT port, uint8_t index>
	void Gpio<port, index>::make_input(void)
	{
		using namespace GPIO;
		switch (port)
		{
			case PORT::B: CLEAR(DDRB, index); break;
			case PORT::C: CLEAR(DDRC, index); break;
			case PORT::D: CLEAR(DDRD, index); break;
		}
	}

	template <GPIO::PORT port, uint8_t index>
	bool Gpio<port, index>::is_high(void) const
	{
		using namespace GPIO;
		switch (port)
		{
			case PORT::B: return IS_SET(PINB, index);
			case PORT::C: return IS_SET(PINC, index);
			case PORT::D: return IS_SET(PIND, index);
		}
	}
	template <GPIO::PORT port, uint8_t index>
	bool Gpio<port, index>::is_low(void) const
	{
		using namespace GPIO;
		switch (port)
		{
			case PORT::B: return IS_CLEAR(PINB, index);
			case PORT::C: return IS_CLEAR(PINC, index);
			case PORT::D: return IS_CLEAR(PIND, index);
		}
	}

	template <GPIO::PORT port, uint8_t index>
	bool Gpio<port, index>::is_output(void) const
	{
		using namespace GPIO;
		switch (port)
		{
			case PORT::B: return IS_SET(DDRB, index);
			case PORT::C: return IS_SET(DDRC, index);
			case PORT::D: return IS_SET(DDRD, index);
		}
	}
	template <GPIO::PORT port, uint8_t index>
	bool Gpio<port, index>::is_input(void) const
	{
		using namespace GPIO;
		switch (port)
		{
			case PORT::B: return IS_CLEAR(DDRB, index);
			case PORT::C: return IS_CLEAR(DDRC, index);
			case PORT::D: return IS_CLEAR(DDRD, index);
		}
	}

	template <GPIO::PORT port, uint8_t index>
	void Gpio<port, index>::pull_up(void)
	{
		using namespace GPIO;
		switch (port)
		{
			case PORT::B: SET(PORTB, index);
			case PORT::C: SET(PORTC, index);
			case PORT::D: SET(PORTD, index);
		}
	}
	template <GPIO::PORT port, uint8_t index>
	void Gpio<port, index>::disconnect(void)
	{
		using namespace GPIO;
		switch (port)
		{
			case PORT::B: CLEAR(PORTB, index);
			case PORT::C: CLEAR(PORTC, index);
			case PORT::D: CLEAR(PORTD, index);
		}
	}
	template <GPIO::PORT port, uint8_t index>
	bool Gpio<port, index>::is_pulled_up(void) const
	{
		using namespace GPIO;
		switch (port)
		{
			case PORT::B: return IS_SET(PORTB, index);
			case PORT::C: return IS_SET(PORTC, index);
			case PORT::D: return IS_SET(PORTD, index);
		}
	}
	template <GPIO::PORT port, uint8_t index>
	bool Gpio<port, index>::is_disconnected(void) const
	{
		using namespace GPIO;
		switch (port)
		{
			case PORT::B: return IS_CLEAR(PORTB, index);
			case PORT::C: return IS_CLEAR(PORTC, index);
			case PORT::D: return IS_CLEAR(PORTD, index);
		}
	}
}

#endif
