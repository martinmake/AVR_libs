#include <avr/io.h>
#include <util/delay.h>

#include <standard/standard.h>

Bit led_port({PORTB, PB5});
Bit led_dd  ({DDRB , DD5});

void init(void)
{
	led_dd.set();
}

int main(void)
{
	init();

	while (1) {
		led_port.set();
		_delay_ms(250);
		led_port.clear();
		_delay_ms(250);
	}
}
