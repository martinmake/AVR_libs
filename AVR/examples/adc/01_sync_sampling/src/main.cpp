#include <avr/io.h>

#include <util.h>
#include <adc.h>
#include <usart/usart0.h>

void init(void)
{
	adc.init({ });
	adc.channel(5);

	usart0.init({ TIO_BAUD });
	stdout = usart0.stream();

	sei();
}

int main(void)
{
	init();

	while (1)
	{
		printf("%u\n", adc.take_sample());
	}
}
