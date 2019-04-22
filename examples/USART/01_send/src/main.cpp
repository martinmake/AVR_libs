#include <avr/io.h>
#include <util/delay.h>
#include <stdio.h>

#include <usart/usart0.h>

Usart0 tio(TIO_BAUD, F_CPU);

void init(void)
{
}

int main(void)
{
	init();

	while (1) {
		tio << 'X';
		tio << '\n';
		tio << "TEST!";
		tio << '\n';
		tio.sendf(10, "2 + 2 = %d", 2 + 2);
		tio << '\n';
		_delay_ms(1000);
	}
}
