#include <usart/usart0.h>
#include <adc.h>
#include <servo/timer1/all.h>

using namespace Usart;

Servo::Timer1::A servo1;
Servo::Timer1::B servo2;

void init(void)
{
	sei();

	usart0.init({ TIO_BAUD });
	stdout = usart0.stream();

	servo1.init();
	servo2.init();

	servo1.pulse_width(1500);
	servo1.pulse_width(2000);

	adc.init({ });
	adc.channel(0);
	adc.sample();

	adc.start_sampling();
}

int main(void)
{
	init();

	while (true)
	{
		printf("%04u\n", adc.value);
		servo1.pulse_width(1000 + adc.value);
		servo1.pulse_width(1000 + adc.value);
	}
}
