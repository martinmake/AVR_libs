#include <system_clock.h>
#include <usart/usart0.h>
#include <led.h>

using namespace Usart;

Led<PORT::B, 1, POLARITY::INVERTED> led;

void init(void)
{
	system_clock.init({ SystemClock::TIMER::TIMER0 });

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
