#include <adc.h>
#include <math/pid/float.h>
#include <usart/usart0.h>

using namespace Usart;

#define KP       1.0
#define KI       0.1
#define KD       1.0
#define LIMIT 1000.0

Pid::Float pid(KP, KI, KD, LIMIT);

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
		printf("DESIRED %+08.2f | ACTUAL %+08.2f | CONTROL %+08.2f\n",
			0.0, adc.value - 512.0, pid(-1 * (adc.value - 512.0)));
	}
}
