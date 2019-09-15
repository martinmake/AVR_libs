#include <avr/interrupt.h>

#include <usart/usart0.h>

void init(void)
{
	usart0.init({ TIO_BAUD });
	stdout = usart0.stream();

	sei();
}

int main(void)
{
	init();

	while (true)
	{
		usart0 << "ABCDEF" << '\n';
		usart0 << 'X' << '\n' << "TEST" << '\n';
		printf("2 + 2 = %d\n", 2 + 2);
	}
}
