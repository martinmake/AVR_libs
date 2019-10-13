#include <gpio.h>

Gpio<PORT::B, 5> led_pin(OUTPUT);

int main(void)
{
	while (true)
	{
		led_pin.toggle();
		_delay_ms(500);
	}
}
