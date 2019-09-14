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
	adc.ISR_callback = [](uint16_t result)
	{
		led.toggle();
		printf("%u\n", result);
	};

	sei();

	adc.start_sampling();
}

volatile bool adc_flag = 0;
int main(void)
{
	init();

	while (1)
	{
		if (adc_flag)
		{
			printf("CONVERSION COMPLETE\n");
			adc_flag = 0;
		}

		// if (ADCSRA & BIT(ADIF)) printf("ADIF IS SET\n");
		// else                    printf("ADIF IS CLEARED\n");
	}
}

ISR(ADC_vect)
{
	adc_flag = 1;
}
