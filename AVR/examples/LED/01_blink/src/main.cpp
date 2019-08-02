#include <avr/io.h>
#include <util/delay.h>

#include <led/led.h>

Led led({{PORTB, PB1}}, true);

int main(void)
{
	while (1) {
		led = 1;
		_delay_ms(100);
		led = 0;
		_delay_ms(100);
	}
}
