#ifndef _USART_USART_H_
#define _USART_USART_H_

#include "iusart.h"

class Usart
{
	public:
		Usart(const Init* init);
		Usart(uint32_t baud, uint32_t f_osc);

		inline Usart& operator<<(const char* s)
		{
			while (*s)
				*this << *s++;

			return *this;
		}
		inline Usart& operator<<(char c)
		{
			output_queue << c;

 		 	if (UCSRA & (1 << UDRE))
 		 		output_queue >> UDR;

			return *this;
		}
};

extern Usart usart;

#endif
