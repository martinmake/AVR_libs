#include <avr/io.h>
#include <inttypes.h>

#include <led/led.h>
#include <usart/usart0.h>

enum class Command : char
{
	OFF = '0',
	ON  = '1'
};

Led<Port::B, 5> led;
Usart0 usart0(TIO_BAUD, F_CPU);

int main(void)
{
	while (1) switch (static_cast<Command>(usart0.getc()))
	{
		case Command::OFF: led = OFF; break;
		case Command::ON:  led = ON;  break;
	}
}
