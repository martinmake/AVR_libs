#include <avr/io.h>
#include <stdio.h>

#include <standard/standard.h>

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
			Bit(UCSR0A, U2X0).set();
		else if (init->x2 == X2::OFF)
			Bit(UCSR0A, U2X0).clear();

		if (init->rx == RX::ON)
			Bit(UCSR0B, RXEN0).set();
		else if (init->rx == RX::OFF)
			Bit(UCSR0B, RXEN0).clear();

		if (init->tx == TX::ON)
			Bit(UCSR0B, TXEN0).set();
		else if (init->tx == TX::OFF)
			Bit(UCSR0B, TXEN0).clear();

		if (init->stop_bit_select == STOP_BIT_SELECT::S1)
			Bit(UCSR0C, USBS0).set();
		else if (init->stop_bit_select == STOP_BIT_SELECT::S2)
			Bit(UCSR0C, USBS0).clear();

		uint8_t character_size = static_cast<uint8_t>(init->character_size);
		Bit(UCSR0C, UCSZ00).write(character_size & 0b001);
		Bit(UCSR0C, UCSZ01).write(character_size & 0b010);
		Bit(UCSR0B, UCSZ02).write(character_size & 0b100);
	}
#elif defined(__AVR_ATmega16__) || defined(__AVR_ATmega16L__)
	void begin(const INIT* init)
	{
		uint16_t brrv;

		brrv  = init->f_osc/(init->x2 == X2::ON ? 8 : 16)/init->baud-1;
		UBRRH = (brrv >> 8);
		UBRRL = (brrv & 0xff);

		if (init->x2 == X2::ON)
			Bit(UCSRA, U2X).set();
		else if (init->x2 == X2::OFF)
			Bit(UCSRA, U2X).clear();

		if (init->rx == RX::ON)
			Bit(UCSRB, RXEN).set();
		else if (init->rx == RX::OFF)
			Bit(UCSRB, RXEN).clear();

		if (init->tx == TX::ON)
			Bit(UCSRB, TXEN).set();
		else if (init->tx == TX::OFF)
			Bit(UCSRB, TXEN).clear();

		UCSRC |= (1 << URSEL);

		if (init->stop_bit_select == STOP_BIT_SELECT::S1)
			Bit(UCSRC, USBS).set();
		else if (init->stop_bit_select == STOP_BIT_SELECT::S2)
			Bit(UCSRC, USBS).clear();

		uint8_t character_size = static_cast<uint8_t>(init->character_size);
		Bit(UCSRC, UCSZ0).write(character_size & 0b001);
		Bit(UCSRC, UCSZ1).write(character_size & 0b010);
		Bit(UCSRB, UCSZ2).write(character_size & 0b100);
	}
#endif

	void begin(uint32_t baud, uint32_t f_osc)
	{
			INIT init;
			init.x2 = X2::OFF;
			init.rx = RX::ON;
			init.tx = TX::ON;
			init.character_size  = CHARACTER_SIZE::S8;
			init.stop_bit_select = STOP_BIT_SELECT::S2;
			init.baud  = baud;
			init.f_osc = f_osc;

			begin(&init);
	}

#if defined(__AVR_ATmega48P__) || defined(__AVR_ATmega88P__) || defined(__AVR_ATmega168P__) || defined(__AVR_ATmega328P__)
	void sendc(char c)
	{
		while (!(UCSR0A & (1 << UDRE0)))
			;

		UDR0 = c;
	}
#elif defined(__AVR_ATmega16__) || defined(__AVR_ATmega16L__)
	void sendc(char c)
	{
		while (!(UCSRA & (1 << UDRE)))
			;

		UDR = c;
	}
#endif

	void sends(const char* str)
	{
		while (*str != '\0')
			sendc(*str++);
	}

	void sendf(size_t size, const char* format, ...)
	{
		va_list ap;
		char* buf = new char[size];

		va_start(ap, format);
		vsnprintf(buf, size, format, ap);
		va_end(ap);

		sends(buf);

		delete[] buf;
	}

#if defined(__AVR_ATmega48P__) || defined(__AVR_ATmega88P__) || defined(__AVR_ATmega168P__) || defined(__AVR_ATmega328P__)
	char recvc()
	{
		while (!(UCSR0A & (1 << RXC0)))
			;

		return UDR0;
	}
#elif defined(__AVR_ATmega16__) || defined(__AVR_ATmega16L__)
	char recvc()
	{
		while (!(UCSRA & (1 << RXC)))
			;

		return UDR;
	}
#endif
}
