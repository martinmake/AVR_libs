#include <avr/io.h>
#include <inttypes.h>

#include <led.h>
#include <usart/usart0.h>

enum class Command : char
{
	OFF = '0',
	ON  = '1'
};

Led<PORT::B, 5> led;

void init(void)
{
	Usart0::Init usart0_init;
	usart0_init.baud = TIO_BAUD;
	usart0 = Usart0(usart0_init);
}

int main(void)
{
	init();

	while (1) switch (static_cast<Command>(usart0.getchar()))
	{
		case Command::OFF: led = OFF; break;
		case Command::ON:  led = ON;  break;
	}
}
