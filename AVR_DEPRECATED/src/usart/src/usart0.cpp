#if defined(__AVR_ATmega48P__) || defined(__AVR_ATmega88P__) || defined(__AVR_ATmega168P__) || defined(__AVR_ATmega328P__)

#include "config.h"
#include "usart0.h"

Usart0::Usart0(const Init* init)
{
	uint16_t brrv = 0;

	switch (init->x2) {
		case X2::ON:
			Bit(UCSR0A, U2X0).set();
			brrv = init->f_osc/8/init->baud-1;
			break;
		case X2::OFF:
			Bit(UCSR0A, U2X0).clear();
			brrv = init->f_osc/16/init->baud-1;
			break;
	}

	UBRR0H = (brrv >> 8);
	UBRR0L = (brrv & 0xff);

	switch (init->rx) {
		case Rx::ON:
			Bit(UCSR0B, RXEN0).set();
			break;
		case Rx::OFF:
			Bit(UCSR0B, RXEN0).clear();
			break;
	}

	switch (init->tx) {
		case Tx::ON:
			Bit(UCSR0B, TXEN0).set();
			break;
		case Tx::OFF:
			Bit(UCSR0B, TXEN0).clear();
			break;
	}

	switch (init->stop_bit_select) {
		case StopBitSelect::S1:
			Bit(UCSR0C, USBS0).set();
			break;
		case StopBitSelect::S2:
			Bit(UCSR0C, USBS0).clear();
			break;
	}

	switch (init->character_size) {
		case CharacterSize::S5:
			Bit(UCSR0C, UCSZ00).clear();
			Bit(UCSR0C, UCSZ01).clear();
			Bit(UCSR0B, UCSZ02).clear();
			break;
		case CharacterSize::S6:
			Bit(UCSR0C, UCSZ00).set();
			Bit(UCSR0C, UCSZ01).clear();
			Bit(UCSR0B, UCSZ02).clear();
			break;
		case CharacterSize::S7:
			Bit(UCSR0C, UCSZ00).clear();
			Bit(UCSR0C, UCSZ01).set();
			Bit(UCSR0B, UCSZ02).clear();
			break;
		case CharacterSize::S8:
			Bit(UCSR0C, UCSZ00).clear();
			Bit(UCSR0C, UCSZ01).set();
			Bit(UCSR0B, UCSZ02).set();
			break;
		case CharacterSize::S9:
			Bit(UCSR0C, UCSZ00).set();
			Bit(UCSR0C, UCSZ01).set();
			Bit(UCSR0B, UCSZ02).set();
			break;
	}

	UCSR0B |= (1 << UDRIE0);
	output_queue = Queue(init->output_queue_size);
}

Usart0::Usart0(uint32_t baud, uint32_t f_osc)
{
		Init init;

		init.x2                = X2::OFF;
		init.rx                = Rx::ON;
		init.tx                = Tx::ON;
		init.character_size    = CharacterSize::S8;
		init.stop_bit_select   = StopBitSelect::S1;
		init.baud              = baud;
		init.f_osc             = f_osc;
		init.output_queue_size = DEFAULT_OUTPUT_QUEUE_SIZE;

		*this = Usart0(&init);
}

ISR(USART_UDRE_vect)
{
	usart0.output_queue >> UDR0;
}

#endif
