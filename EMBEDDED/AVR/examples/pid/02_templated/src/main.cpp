#include <adc.h>
#include <math/pid/templated.h>
#include <usart/usart0.h>

using namespace Usart;

#define KP 4
#define KI 1
#define KD 8
#define INTEGRAL_LIMIT 30

Pid::Templated<int16_t, int16_t, uint8_t, KP, KI, KD, INTEGRAL_LIMIT> pid;

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
		printf("DESIRED %+04d | ACTUAL %+04d | CONTROL %03u\n",
			0, adc.value - 512, pid(-1 * (adc.value - 512)));
	}
}
