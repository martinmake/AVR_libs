#ifndef _USART_IUSART_H_
#define _USART_IUSART_H_

#include <stdio.h>

#include "queue.h"

class IUsart
{
	public: // TYPES
		enum class X2            : uint8_t { OFF, ON };
		enum class Rx            : uint8_t { OFF, ON };
		enum class Tx            : uint8_t { OFF, ON };
		enum class StopBitSelect : uint8_t { S1, S2 };
		enum class CharacterSize : uint8_t { S5, S6, S7, S8, S9 };

		struct Init
		{
			X2            x2;
			Rx            rx;
			Tx            tx;
			CharacterSize character_size;
			StopBitSelect stop_bit_select;
			uint32_t      baud;
			uint32_t      f_osc;
			uint8_t       output_queue_size;
		};

	public: // CONSTRUCTORS
		IUsart();

	public: // FUNCTIONS
	//	static inline int putc   (char var, FILE* stream);

	//	virtual inline IUsart& operator<<(const char* s) = 0;
	//	virtual inline IUsart& operator<<(      char  c) = 0;

	public:
		Queue output_queue;
};

#endif
