#include <avr/io.h>

#include <standard/standard.h>
#include <usart/usart0.h>

Usart0 usart0(TIO_BAUD, F_CPU);

Pin echo(PORTD, PD2);
Pin trig(PORTD, PD3);

void init(void)
{
	trig.dd.set();

	/* # COUNTER # */
	TCCR1B |= (1 << CS12) | (0 << CS11) | (1 << CS10);
	/* # COUNTER # */
}

int main(void)
{
	init();

	while (1) {
		trig.port.set();
		trig.port.clear();

		uint16_t duration = 0;
		uint16_t distance = duration * 340 / 2;

		usart0.sendf(10, "% 4dm\n", distance);
	}
}
