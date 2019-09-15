#include <avr/io.h>

#define UTIL_DEFINE_SLEEP
#include <util.h>
#include <adc.h>
#include <led.h>
#include <usart/usart0.h>

Adc adc;
Led<PORT::B, 5> led;
Usart0 usart0(TIO_BAUD, F_CPU);

void init(void)
{
	stdout = usart0.stream();

	adc.channel(0);
	adc.on_conversion = [](uint16_t result)
	{
		printf("%u\n", result);
		usart0.flush();
	};

	sei();

	adc.start_sampling();
}

int main(void)
{
	init();
	while (true) {}
}
