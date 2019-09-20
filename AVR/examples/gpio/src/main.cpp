#define UTIL_DEFINE_SLEEP
#include <util.h>
#include <gpio.h>

Gpio<PORT::B, 5> led;

inline void init(void)
{
}

int main(void)
{
	init();

	while (true)
	{
		led = HIGH;
		_delay_ms(200);
		led = LOW;
		_delay_ms(200);
	}
}
