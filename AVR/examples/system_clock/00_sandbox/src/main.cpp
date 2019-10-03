#include <avr/io.h>

#include <system_clock/timer0.h>
#include <usart/usart0.h>

using namespace Usart;
using namespace SystemClock;

void init(void)
{
	usart0.init({ TIO_BAUD });
	stdout = usart0.stream();

	sei();
}

int main(void)
{
	init();

	while (true)
		printf("%lu\n", system_clock.time());
}
