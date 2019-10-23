#include <core/gpio.h>

using namespace Core;
using namespace GPIO;

Gpio<PORT::B, 5> led_pin(MODE::OUTPUT);

int main(void)
{
	while (true)
	{
		led_pin.toggle();
		_delay_ms(500);
	}
}
