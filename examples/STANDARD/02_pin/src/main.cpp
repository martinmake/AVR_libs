#include <avr/io.h>
#include <util/delay.h>

#include <standard/standard.h>
#include <usart/usart.h>

Pin led(PORTB, PB5);

void init(void)
{
	led.dd.set();
}

int main(void)
{
	init();

	while (1) {
		led.port.set();
		_delay_ms(250);
		led.port.clear();
		_delay_ms(250);
	}
}
