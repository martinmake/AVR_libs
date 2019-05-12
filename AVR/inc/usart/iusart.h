#ifndef _USART_IUSART_H_
#define _USART_IUSART_H_

#include <avr/interrupt.h>
#include <avr/io.h>

#include <standard/standard.h>

#include "queue.h"

class IUsart
{
	public:
		enum class X2 : uint8_t {
			OFF, ON
		};

		enum class Rx : uint8_t {
			OFF, ON
		};

		enum class Tx : uint8_t {
			OFF, ON
		};

		enum class StopBitSelect : uint8_t {
			S1, S2
		};

		enum class CharacterSize : uint8_t {
			S5, S6, S7, S8, S9
		};

		struct Init {
			X2            x2;
			Rx            rx;
			Tx            tx;
			CharacterSize character_size;
			StopBitSelect stop_bit_select;
			uint32_t      baud;
			uint32_t      f_osc;
			uint8_t       output_queue_size;
		};

	public:
		Queue output_queue;

	public:
		IUsart() {};

		void sendf(uint16_t size, const char* format, ...);

		virtual inline IUsart& operator<<(const char* s) { (void)s; return *this; };
		virtual inline IUsart& operator<<(char c)        { (void)c; return *this; };
};

#endif
