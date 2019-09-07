#if defined(__AVR_ATmega16__) || defined(__AVR_ATmega16L__)

#include "config.h"
#include "usart.h"

void Usart::Usart(const Init* init)
{
	uint16_t brrv = 0;

	switch (init->x2) {
		case X2::ON:
			Bit(UCSRA, U2X).set();
			brrv = init->f_osc/8/init->baud-1;
			break;
		case X2::OFF:
			Bit(UCSRA, U2X).clear();
			brrv = init->f_osc/16/init->baud-1;
			break;
	}

	UBRRH = (brrv >> 8);
	UBRRL = (brrv & 0xff);

	switch (init->rx) {
		case Rx::ON:
			Bit(UCSRB, RXEN).set();
			break;
		case Rx::OFF:
			Bit(UCSRB, RXEN).clear();
			break;
	}

	switch (init->tx) {
		case Tx::ON:
			Bit(UCSRB, TXEN).set();
			break;
		case Tx::OFF:
			Bit(UCSRB, TXEN).clear();
			break;
	}

	switch (init->stop_bit_select) {
		case StopBitSelect::S1:
			Bit(UCSRC, USBS).set();
			break;
		case StopBitSelect::S2:
			Bit(UCSRC, USBS).clear();
			break;
	}

	switch (init->character_size) {
		case CharacterSize::S5:
			Bit(UCSRC, UCSZ0).clear();
			Bit(UCSRC, UCSZ1).clear();
			Bit(UCSRB, UCSZ2).clear();
			break;
		case CharacterSize::S6:
			Bit(UCSRC, UCSZ0).set();
			Bit(UCSRC, UCSZ1).clear();
			Bit(UCSRB, UCSZ2).clear();
			break;
		case CharacterSize::S7:
			Bit(UCSRC, UCSZ0).clear();
			Bit(UCSRC, UCSZ1).set();
			Bit(UCSRB, UCSZ2).clear();
			break;
		case CharacterSize::S8:
			Bit(UCSRC, UCSZ0).clear();
			Bit(UCSRC, UCSZ1).set();
			Bit(UCSRB, UCSZ2).set();
			break;
		case CharacterSize::S9:
			Bit(UCSRC, UCSZ0).set();
			Bit(UCSRC, UCSZ1).set();
			Bit(UCSRB, UCSZ2).set();
			break;
	}

	UCSRB |= (1 << TXCIE);
	output_queue = Queue(init->output_queue_size);
}

Usart::Usart(uint32_t baud, uint32_t f_osc)
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

		*this = Usart(&init);
}

ISR(USART_TX_vect)
{
	while ( ! (UCSRA & (1 << UDRE)) ) {}
	usart.output_queue >> UDR;
}

#endif
