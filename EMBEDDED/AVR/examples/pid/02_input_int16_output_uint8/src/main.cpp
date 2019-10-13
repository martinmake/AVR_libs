#include <adc.h>
#include <math/pid.h>
#include <usart/usart0.h>

using namespace Usart;

Pid<int16_t, int16_t, uint8_t, 1, 1, 1, 255> pid;

void init(void)
{
	usart0.init({ TIO_BAUD });
	stdout = usart0.stream();

	adc.init({ });
	adc.channel(0);

	sei();

	adc.start_sampling();
}

int main(void)
{
	init();

	while (true)
	{
		printf("DESIRED %4d | ACTUAL %4d | CONTROL %4d\n", 0, adc.value - 512, pid(0, adc.value - 512));
	}
}
