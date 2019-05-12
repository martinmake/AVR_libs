#include <avr/io.h>
#include <util/delay.h>
#include <inttypes.h>

#include <led/led.h>
#include <usart/usart0.h>

enum Command : char {
	OFF = '0',
	ON  = '1'
};

Led    led({PORTB, PB5});
Usart0 usart0(TIO_BAUD, F_CPU);

int main(void)
{
	while (1) {
		char command;
		usart0 >> command;

		switch (command) {
			case Command::OFF: led = 0; break;
			case Command::ON:  led = 1; break;
		}
	}
}
