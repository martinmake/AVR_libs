#include <avr/io.h>
#include <stdio.h>

#include "usart.h"

namespace Usart
{
#if defined(__AVR_ATmega48P__) || defined(__AVR_ATmega88P__) || defined(__AVR_ATmega168P__) || defined(__AVR_ATmega328P__)
	void begin(const INIT* init)
	{
		uint16_t brrv;

		brrv   = init->f_osc/(init->x2 == X2::ON ? 8 : 16)/init->baud-1;
		UBRR0H = (brrv >> 8);
		UBRR0L = (brrv & 0xff);

		if (init->x2 == X2::ON)
			UCSR0A |= (1 << U2X0);

		if (init->rx == RX::ON)
			UCSR0B |= (1 << RXEN0);
		if (init->tx == TX::ON)
			UCSR0B |= (1 << TXEN0);

		if (init->stop_bit_select == STOP_BIT_SELECT::TWO)
			UCSR0C |= (1 << USBS0);

		uint8_t character_size = static_cast<uint8_t>(init->character_size);
		UCSR0C |= (character_size & 0b001 << UCSZ00);
		UCSR0C |= (character_size & 0b010 << UCSZ01);
		UCSR0B |= (character_size & 0b100 << UCSZ02);
	}
#elif defined(__AVR_ATmega16__) || defined(__AVR_ATmega16L__)
	void begin(const INIT* init)
	{
		uint16_t brrv;

		brrv  = init->f_osc/(init->x2 == X2::ON ? 8 : 16)/init->baud-1;
		UBRRH = (brrv >> 8);
		UBRRL = (brrv & 0xff);

		if (init->x2 == X2::ON)
			UCSRA |= (1 << U2X);

		if (init->rx == RX::ON)
			UCSRB |= (1 << RXEN);
		if (init->tx == TX::ON)
			UCSRB |= (1 << TXEN);

		UCSRC |= (1 << URSEL);

		if (init->stop_bit_select == STOP_BIT_SELECT::TWO)
			UCSRC |= (1 << USBS);

		uint8_t character_size = static_cast<uint8_t>(init->character_size);
		UCSRC |= (character_size & 0b001 << UCSZ0);
		UCSRC |= (character_size & 0b010 << UCSZ1);
		UCSRB |= (character_size & 0b100 << UCSZ2);
	}
#endif

#if defined(__AVR_ATmega48P__) || defined(__AVR_ATmega88P__) || defined(__AVR_ATmega168P__) || defined(__AVR_ATmega328P__)
	void begin_simple(uint32_t baud, uint32_t f_osc)
	{
		uint16_t brrv;

		brrv  = f_osc/16/baud-1;

		UBRR0H = (brrv >> 8);
		UBRR0L = (brrv & 0xff);

		UCSR0A &= ~(1 << U2X0);
		UCSR0B |=  (1 << RXEN0) | (1 << TXEN0);
		UCSR0C |=  (1 << USBS0);

		uint8_t character_size = static_cast<uint8_t>(Usart::CHARACTER_SIZE::EIGHT);
		UCSR0C |= (character_size & 0b001 << UCSZ00);
		UCSR0C |= (character_size & 0b010 << UCSZ01);
		UCSR0B |= (character_size & 0b100 << UCSZ02);
	}
#elif defined(__AVR_ATmega16__) || defined(__AVR_ATmega16L__)
	void begin_simple(uint32_t baud, uint32_t f_osc)
	{
		uint16_t brrv;

		brrv  = f_osc/16/baud-1;
		UBRRH = (brrv >> 8);
		UBRRL = (brrv & 0xff);

		UCSRA &= ~(1 << U2X);
		UCSRB |=  (1 << RXEN) | (1 << TXEN);;

		UCSRC |=  (1 << URSEL);
		UCSRC |=  (1 << USBS);

		uint8_t character_size = static_cast<uint8_t>(Usart::CHARACTER_SIZE::EIGHT);
		UCSRC |= (character_size & 0b001 << UCSZ0);
		UCSRC |= (character_size & 0b010 << UCSZ1);
		UCSRB |= (character_size & 0b100 << UCSZ2);
	}
#endif

#if defined(__AVR_ATmega48P__) || defined(__AVR_ATmega88P__) || defined(__AVR_ATmega168P__) || defined(__AVR_ATmega328P__)
	void send_char(char c)
	{
		while (!(UCSR0A & (1 << UDRE0)))
			;

		UDR0 = c;
	}
#elif defined(__AVR_ATmega16__) || defined(__AVR_ATmega16L__)
	void send_char(char c)
	{
		while (!(UCSRA & (1 << UDRE)))
			;

		UDR = c;
	}
#endif

	void send_bits(uint8_t bits, NL nl)
	{
		uint8_t mask = (1 << 7);

		while (mask) {
			if (bits & mask)
				send_char('1');
			else
				send_char('0');

			if (mask == 0b00010000)
				send_char(' ');

			mask >>= 1;
		}

		if (nl == NL::ON)
			send_char('\n');
	}

	void send_str(const char* str)
	{
		while (*str != '\0')
			send_char(*str++);
	}

	void sendf(size_t size, const char* format, ...)
	{
		va_list ap;
		char* buf = new char[size];

		va_start(ap, format);
		vsnprintf(buf, size, format, ap);
		va_end(ap);

		send_str(buf);

		delete[] buf;
	}
}
