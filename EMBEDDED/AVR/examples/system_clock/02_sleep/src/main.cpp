#include <system_clock.h>
#include <led.h>

Led<PORT::B, 5> led;

void init(void)
{
	system_clock.init({ SystemClock::TIMER::TIMER0 });

	sei();
}

int main(void)
{
	init();

	while (true)
	{
		led.toggle();
		system_clock.sleep(200);
	}
}
