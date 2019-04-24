#include <avr/io.h>
#include <util/delay.h>

#include <standard/standard.h>
#include <led/led.h>

Led led_err({PORTB, PB5});
Led led_ext({PORTB, PB4}, true);

int main(void)
{
	while (1) {
		led_ext = led_err = 1;
		_delay_ms(100);
		led_ext = led_err = 0;
		_delay_ms(100);
	}
}
