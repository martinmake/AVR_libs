#ifndef _USART_USART0_H_
#define _USART_USART0_H_

#include <avr/io.h>
#include <stdio.h>

#include <standard/standard.h>

extern int sendchar(char c, FILE* stream);

class Usart0
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
			S5 = 0b000,
			S6 = 0b001,
			S7 = 0b010,
			S8 = 0b011,
			S9 = 0b111
		};

		typedef struct {
			X2            x2;
			Rx            rx;
			Tx            tx;
			CharacterSize character_size;
			StopBitSelect stop_bit_select;
			uint32_t      baud;
			uint32_t      f_osc;
		} INIT;

	public:
		Usart0(const INIT* init);
		Usart0(uint32_t baud, uint32_t f_osc);

		void sendf(uint16_t size, const char* format, ...);
		inline void operator<<(const char* s) { while (*s != '\0') *this << *s++;       }
		inline void operator<<(char  c) { while (!(UCSR0A & (1 << UDRE0))) {} UDR0 = c; }
		inline void operator>>(char& c) { while (!(UCSR0A & (1 << RXC0)))  {} c = UDR0; }
};

#endif
