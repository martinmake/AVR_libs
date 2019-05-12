#ifndef _USART_USART0_H_
#define _USART_USART0_H_

#include "iusart.h"

class Usart0: virtual public IUsart
{
	public:
		Usart0(const Init* init);
		Usart0(uint32_t baud, uint32_t f_osc);

		inline Usart0& operator<<(char c)
		{
			output_queue << c;

 		 	if (UCSR0A & (1 << UDRE0))
 		 		output_queue >> UDR0;

			return *this;
		}
		inline Usart0& operator<<(const char* s)
		{
			while (*s)
				*this << *s++;

			return *this;
		}

		inline Usart0& operator>>(char& c)
		{
			while ( ! (UCSR0A & (1 << RXC0)) ) {}

			c = UDR0;

			return *this;
		}
		inline Usart0& operator>>(char* s)
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
};

extern Usart0 usart0;

#endif
