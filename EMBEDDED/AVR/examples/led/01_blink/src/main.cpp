#include <avr/io.h>
#include <util/delay.h>

#include <led.h>

Led<PORT::B, 5>  led;

int main(void)
{
	while (1)
	{
		led.toggle();
		_delay_ms(100);
	}
}
