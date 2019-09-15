#include <avr/io.h>

#include <util.h>
#include <adc.h>
#include <led.h>
#include <usart/usart0.h>

Led<PORT::D, 6, POLARITY::INVERTED> led;

void init(void)
{
	adc.init({ });
	adc.channel(5);
	adc.on_conversion = [](uint16_t result)
	{
		printf("%u\n", result);
		usart0.flush();
	};

	usart0.init({ TIO_BAUD });
	stdout = usart0.stream();

	sei();

	adc.start_sampling();
}

int main(void)
{
	init();

	putchar('\0');
	putchar('\0');

	while (true) {}
}
