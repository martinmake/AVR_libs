#if defined(__AVR_ATmega48P__) || defined(__AVR_ATmega88P__) || defined(__AVR_ATmega168P__) || defined(__AVR_ATmega328P__)

#include <stdio.h>

#define F_CPU 16000000 // TMP
#include <util/delay.h>

#include "usart0.h"
#include "queue.h"

Usart0::Usart0(const INIT* init)
{
	uint16_t brrv;

	brrv   = init->f_osc/(init->x2 == X2::ON ? 8 : 16)/init->baud-1;
	UBRR0H = (brrv >> 8);
	UBRR0L = (brrv & 0xff);

	if (init->x2 == X2::ON)
		Bit(UCSR0A, U2X0).set();
	else if (init->x2 == X2::OFF)
		Bit(UCSR0A, U2X0).clear();

	if (init->rx == Rx::ON)
		Bit(UCSR0B, RXEN0).set();
	else if (init->rx == Rx::OFF) Bit(UCSR0B, RXEN0).clear();
	if (init->tx == Tx::ON)
		Bit(UCSR0B, TXEN0).set();
	else if (init->tx == Tx::OFF)
		Bit(UCSR0B, TXEN0).clear();

	if (init->stop_bit_select == StopBitSelect::S1)
		Bit(UCSR0C, USBS0).set();
	else if (init->stop_bit_select == StopBitSelect::S2)
		Bit(UCSR0C, USBS0).clear();

	uint8_t character_size = static_cast<uint8_t>(init->character_size);
	Bit(UCSR0C, UCSZ00).write(character_size & 0b001);
	Bit(UCSR0C, UCSZ01).write(character_size & 0b010);
	Bit(UCSR0B, UCSZ02).write(character_size & 0b100);

	UCSR0B |= (1 << TXCIE0);
	output_queue = Queue(init->output_queue_size);
}

Usart0::Usart0(uint32_t baud, uint32_t f_osc)
{
		INIT init;

		init.x2                = X2::OFF;
		init.rx                = Rx::ON;
		init.tx                = Tx::ON;
		init.character_size    = CharacterSize::S8;
		init.stop_bit_select   = StopBitSelect::S1;
		init.baud              = baud;
		init.f_osc             = f_osc;
		init.output_queue_size = 64;

		*this = Usart0(&init);
}

void Usart0::sendf(size_t size, const char* format, ...)
{
	va_list ap;
	char* buf = new char[size];

	va_start(ap, format);
	vsnprintf(buf, size, format, ap);
	va_end(ap);

	*this << buf;

	delete[] buf;
}

ISR(USART_TX_vect)
{
	while ( ! (UCSR0A & (1 << UDRE0)) ) {}
	usart0.output_queue >> UDR0;
}

#endif
