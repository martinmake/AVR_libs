#include <avr/io.h>
#include <util/delay.h>

#include <standard/standard.h>
#include <usart/usart0.h>

Usart0 usart0(TIO_BAUD, F_CPU);

void init(void)
{
	sei();
}

int main(void)
{
	init();

	while (1) {
		usart0 << "ABCDEF" << '\n';
		usart0 << 'X' << '\n' << "TEST" << '\n';
		usart0.sendf(15, "2 + 2 = %d\n", 2 + 2);
	}
}
