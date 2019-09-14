#include <avr/io.h>

#define UTIL_DEFINE_SLEEP
#include <util/util.h>
#include <adc/adc.h>
#include <led/led.h>
#include <usart/usart0.h>

Adc adc;
Led<PORT::B, 5> led;
Usart0 usart0(TIO_BAUD, F_CPU);

void init(void)
{
	stdout = usart0.stream();

	adc.channel(0);
	adc.ISR_callback = [](uint16_t result)
	{
		printf("%u\n", result);
		led.toogle();
	};

	sei();

	// adc.start_sampling();
}

int main(void)
{
	init();

	while (1)
	{
		// led.toogle();
		// sleep(1000);
		printf("%u\n", adc.take_sample());
	}
}
