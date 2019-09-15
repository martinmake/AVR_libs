#ifndef _USART_IUSART_H_
#define _USART_IUSART_H_

#include <stdio.h>
#undef getchar
#undef putchar

#include "usart/queue.h"
#include "usart/config.h"

class IUsart
{
	public: // CONSTRUCTORS
		         IUsart();
		virtual ~IUsart();

	public: // GETTERS
		virtual FILE* stream(void) = 0;

	public: // FUNCTIONS
		virtual void putchar(char c) = 0;
		virtual char getchar(void  ) = 0;
		virtual void flush  (void  ) = 0;

	public:
		Queue output_queue;
};

#endif
