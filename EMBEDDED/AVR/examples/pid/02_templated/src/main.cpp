#include <adc.h>
#include <math/pid/templated.h>
#include <usart/usart0.h>

using namespace Usart;

#define KP 4
#define KI 0.01
#define KD 8
#define INTEGRAL_LIMIT 30
#define FACTOR 100
typedef int16_t input_t;
typedef int32_t intermediate_t;
typedef uint8_t output_t;

Pid::Templated<
	input_t,
	intermediate_t,
	output_t,
	(intermediate_t (KP             * FACTOR)),
	(intermediate_t (KI             * FACTOR)),
	(intermediate_t (KD             * FACTOR)),
	(intermediate_t (INTEGRAL_LIMIT * FACTOR)),
	FACTOR> pid;

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
