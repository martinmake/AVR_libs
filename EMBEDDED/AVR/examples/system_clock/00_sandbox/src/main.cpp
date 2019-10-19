#include <system_clock.h>
#include <usart/usart0.h>
#include <gpio.h>
#include <led.h>

using namespace Usart;

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
	{

		SYSTEM_CLOCK_TIMEOUT(100)
		{
			putchar('A');
			break;
		} putchar('\n');

		if (system_clock.has_timed_out()) putchar('\n');

		system_clock.sleep(1000);
	}
}
