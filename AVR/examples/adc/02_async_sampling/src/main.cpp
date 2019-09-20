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

	adc.start_sampling();
}

int main(void)
{
	init();

	while (true) printf("%u\n", adc.value);
}
