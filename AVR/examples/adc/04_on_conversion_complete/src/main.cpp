#include <avr/io.h>

#include <util.h>
#include <led.h>
#include <adc.h>

volatile bool flag = false;

Led<PORT::D, 5, POLARITY::INVERTED> led1;
Led<PORT::D, 6, POLARITY::INVERTED> led2;

void init(void)
{
	adc.init({ });
	adc.channel(5);
	adc.on_conversion = []() { flag = true; };

	sei();

	adc.start_sampling();
}

int main(void)
{
	init();

	while (true) if (flag)
	{
		flag = false;
		led1.toggle();
	} else  led2.toggle();
}
