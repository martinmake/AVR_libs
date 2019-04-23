#include <avr/io.h>
#include <util/delay.h>
#include <inttypes.h>
#include <string.h>

#include <led/led.h>
#include <usart/usart0.h>

#define BUFFER_SIZE 128

Usart0 usart0(TIO_BAUD, F_CPU);

char buffer[BUFFER_SIZE];

void init(void)
{
	sei();
}

int main(void)
{
	init();

	while (1) {
		usart0 >> buffer;
		usart0 << buffer;
	}
}
