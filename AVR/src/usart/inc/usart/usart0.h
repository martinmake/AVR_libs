#ifndef _USART_USART0_H_
#define _USART_USART0_H_

#include <avr/io.h>

#include <util.h>

#include "usart/iusart.h"

class Usart0: virtual public IUsart
{
	public: // CONSTRUCTORS
		Usart0(const Init* init);
		Usart0(uint32_t baud, uint32_t f_osc);

	public: // GETTERS
		FILE* stream(void);

	public: // FUNSTIONS
		inline char getc(void);
		inline void sendf(const char* fmt, ...);
		inline void flush(void);

	public: // OPERATORS
		inline Usart0& operator<<(      char  c);
		inline Usart0& operator<<(const char* s);
		inline Usart0& operator>>(char& c);
		inline Usart0& operator>>(char* s);
};

inline char Usart0::getc(void)
{
	while ( ! (UCSR0A & BIT(RXC0)) ) {}
	return UDR0;
}
inline void Usart0::sendf(const char* fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	vfprintf(stream(), fmt, args);
	va_end(args);
}
inline void Usart0::flush(void)
{
	uint8_t sreg_save = SREG;
	cli();
	while (!output_queue.is_empty())
	{
		while (IS_CLEAR(UCSR0A, UDRE0)) {}
		output_queue >> UDR0;
	}
	SREG = sreg_save;
}

inline Usart0& Usart0::operator<<(char c)
{
	output_queue << c;

	if (UCSR0A & BIT(UDRE0))
		output_queue >> UDR0;

	return *this;
}
inline Usart0& Usart0::operator<<(const char* s)
{
	while (*s)
		*this << *s++;

	return *this;
}

inline Usart0& Usart0::operator>>(char& c)
{
	while ( ! (UCSR0A & BIT(RXC0)) ) {}

	c = UDR0;

	return *this;
}
inline Usart0& Usart0::operator>>(char* s)
{
	s--;
	do {
		s++;
		*this >> *s;
	} while (*s != '\r');

	*(s + 0) = '\n';
	*(s + 1) = '\0';

	return *this;
}

extern Usart0 usart0;

#endif
