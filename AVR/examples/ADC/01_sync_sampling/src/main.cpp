#include <avr/io.h>

#define UTIL_DEFINE_SLEEP
#include <util/util.h>
#include <adc/adc.h>
#include <usart/usart0.h>

Adc adc;
Usart0 usart0(TIO_BAUD, F_CPU);

void init(void)
{
	stdout = usart0.stream();

	adc.channel(0);

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
