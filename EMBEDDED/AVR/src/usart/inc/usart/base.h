#ifndef _USART_BASE_H_
#define _USART_BASE_H_

#include <stdio.h>
#undef getchar
#undef putchar

#include "usart/queue.h"
#include "usart/config.h"

namespace Usart
{
	class Base
	{
		public: // CONSTRUCTORS
				 Base();
			virtual ~Base();

		public: // GETTERS
			virtual FILE* output_stream(void) = 0;

		public: // FUNCTIONS
			virtual void putchar(char c) = 0;
			virtual char getchar(void  ) = 0;
			virtual void flush  (void  ) = 0;

		public:
			Queue output_queue;
	};
}

#endif
