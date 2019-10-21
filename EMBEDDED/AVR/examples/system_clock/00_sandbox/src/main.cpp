#include <system_clock.h>
#include <usart/usart0.h>

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
		puts("TEST");
		system_clock.sleep(1000);

	}
}
