#if defined(__AVR_ATmega48P__) || defined(__AVR_ATmega88P__) || defined(__AVR_ATmega168P__) || defined(__AVR_ATmega328P__)

#include <avr/interrupt.h>

#include "config.h"
#include "usart/usart0.h"

static inline int usart0_putchar(char var, FILE* stream)
{
	(void) stream;

	usart0 << var;

	return 0;
}
static inline int usart0_getchar(FILE* stream)
{
	(void) stream;

	return usart0.getc();
}

static FILE    stream;
static FILE* p_stream = &stream;

Usart0::Usart0(const Init* init)
{
	uint16_t brrv = 0;

	switch (init->x2)
	{
		case X2::ON:
			UCSR0A |= BIT(U2X0);
			brrv = init->f_osc/8/init->baud-1;
			break;
		case X2::OFF:
			UCSR0A &= ~BIT(U2X0);
			brrv = init->f_osc/16/init->baud-1;
			break;
	}

	UBRR0H = (brrv >> 8);
	UBRR0L = (brrv & 0xff);

	switch (init->rx)
	{
		case Rx::ON:  UCSR0B |=  BIT(RXEN0); break;
		case Rx::OFF: UCSR0B &= ~BIT(RXEN0); break;
	}

	switch (init->tx)
	{
		case Tx::ON:  UCSR0B |=  BIT(TXEN0); break;
		case Tx::OFF: UCSR0B &= ~BIT(TXEN0); break;
	}

	switch (init->stop_bit_select)
	{
		case StopBitSelect::S1: UCSR0C |=  BIT(USBS0); break;
		case StopBitSelect::S2: UCSR0C &= ~BIT(USBS0); break;
	}

	switch (init->character_size)
	{
		case CharacterSize::S5:
			UCSR0C &= ~BIT(UCSZ00);
			UCSR0C &= ~BIT(UCSZ01);
			UCSR0B &= ~BIT(UCSZ02);
			break;
		case CharacterSize::S6:
			UCSR0C |=  BIT(UCSZ00);
			UCSR0C &= ~BIT(UCSZ01);
			UCSR0B &= ~BIT(UCSZ02);
			break;
		case CharacterSize::S7:
			UCSR0C &= ~BIT(UCSZ00);
			UCSR0C |=  BIT(UCSZ01);
			UCSR0B &= ~BIT(UCSZ02);
			break;
		case CharacterSize::S8:
			UCSR0C &= ~BIT(UCSZ00);
			UCSR0C |=  BIT(UCSZ01);
			UCSR0B |=  BIT(UCSZ02);
			break;
		case CharacterSize::S9:
			UCSR0C |=  BIT(UCSZ00);
			UCSR0C |=  BIT(UCSZ01);
			UCSR0B |=  BIT(UCSZ02);
			break;
	}

	UCSR0B |= BIT(UDRIE0);
	output_queue = Queue(init->output_queue_size);
	fdev_setup_stream(p_stream, usart0_putchar, usart0_getchar, _FDEV_SETUP_RW);
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

FILE* Usart0::stream(void)
{
	return p_stream;
}

ISR(USART_UDRE_vect) { usart0.output_queue >> UDR0; }

#endif
