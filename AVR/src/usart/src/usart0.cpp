#if defined(__AVR_ATmega48P__) || defined(__AVR_ATmega88P__) || defined(__AVR_ATmega168P__) || defined(__AVR_ATmega328P__)

#include <avr/interrupt.h>

#include <util.h>

#include "usart/usart0.h"

Usart0 usart0;

static inline int usart0_putchar(char var, FILE* stream)
{
	(void) stream;

	usart0.putchar(var);

	return 0;
}
static inline int usart0_getchar(FILE* stream)
{
	(void) stream;

	return usart0.getchar();
}

static FILE    stream;
static FILE* p_stream = &stream;

Usart0::Usart0(void)
{
}
Usart0::Usart0(const Init& init_struct)
{
	init(init_struct);
}

void Usart0::init_pipeline(const Init& init)
{
	output_queue = Queue(init.output_queue_size);
	fdev_setup_stream(p_stream, usart0_putchar, usart0_getchar, _FDEV_SETUP_RW);
}

FILE* Usart0::stream(void) { return p_stream; }

ISR(USART_UDRE_vect) { if (!usart0.output_queue.is_empty()) usart0.output_queue >> UDR0; }

#endif
