#if defined(__AVR_ATmega16__) || defined(__AVR_ATmega16L__)

#include <avr/io.h>
#include <stdio.h>

#include <standard/standard.h>

#include "usart.h"

void Usart::begin(const INIT* init)
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

void Usart::sendc(char c)
{
	while (!(UCSRA & (1 << UDRE)))
		;

	UDR = c;
}

char Usart::recvc()
{
	while (!(UCSRA & (1 << RXC)))
		;

	return UDR;
}

#endif
