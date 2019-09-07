#include <avr/io.h>
#include <avr/interrupt.h>

#include <inttypes.h>
#include <string.h>

#define UTIL_DEFINE_SLEEP
#include <util/util.h>
#include <led/led.h>
#include <usart/usart0.h>

#define BUFFER_SIZE 128

Usart0 usart0(TIO_BAUD, F_CPU);

void init(void)
{
	sei();
}

int main(void)
{
	init();

	while (1)
	{
		char c;
		usart0 >> c;
		usart0 << c;
	}
}
