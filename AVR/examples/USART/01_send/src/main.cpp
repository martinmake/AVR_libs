#include <avr/interrupt.h>

#include <usart/usart0.h>

Usart0 usart0(TIO_BAUD, F_CPU);

void init(void)
{
	stdout = usart0.stream();
	sei();
}

int main(void)
{
	init();

	while (1)
	{
		usart0 << "ABCDEF" << '\n';
		usart0 << 'X' << '\n' << "TEST" << '\n';
		usart0.sendf("2 + 2 = %d\n", 2 + 2);
		printf("2 + 2 = %d\n", 2 + 2);
	}
}
