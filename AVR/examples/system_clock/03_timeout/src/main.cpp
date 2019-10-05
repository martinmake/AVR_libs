#include <system_clock.h>
#include <usart/usart0.h>
#include <gpio.h>
#include <led.h>

using namespace Usart;

#define TIMEOUT_TIME 15000

Led<PORT::B, 1, POLARITY::INVERTED> led;
Gpio<PORT::C, 0> pin(OUTPUT);

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
		printf("YOU HAVE %d SECONDS TO PULL UP PC0 !!\n", TIMEOUT_TIME / 1000);
		bool timed_out = system_clock.timeout(TIMEOUT_TIME, []()
		{
			if (pin) return true;
			else     return false;
		});

		if (timed_out) puts("YOU FAILED !!");
		else           puts("YOU PASSED !!");

		system_clock.sleep(200);
	}
}
