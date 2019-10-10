#include <util.h>
#include <gpio.h>
#include <system_clock.h>
#include <usart/usart0.h>

using namespace Usart;

Gpio<PORT::D, 4> pin_direction;
Gpio<PORT::D, 5> pin_enable;

void init(void)
{
	system_clock.init({ SystemClock::TIMER::TIMER0 });

	usart0.init({ TIO_BAUD });
	stdout = usart0.stream();

	pin_direction = LOW;
	pin_enable    = LOW;

	sei();

	puts("SYSTEM INITIATED!");
}

int main(void)
{
	init();

	system_clock.sleep(1000);
	puts("STARTING MOTOR A");
	pin_direction = LOW;
	pin_enable    = HIGH;
	system_clock.sleep(1000);
	puts("STOPPING MOTOR A");
	pin_direction = LOW;
	pin_enable    = LOW;

	while (true) { }
}
