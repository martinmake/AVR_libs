#include <avr/io.h>
#include <inttypes.h>

#include <led.h>
#include <usart/usart0.h>

using namespace Usart;

enum class COMMAND : char
{
	TURN_LED_OFF = '0',
	TURN_LED_ON  = '1'
};

Led<PORT::B, 5> led;

void init(void)
{
	usart0.init({ TIO_BAUD });
	stdout = usart0.stream();
}

int main(void)
{
	init();

	while (true) switch (static_cast<COMMAND>(getchar()))
	{
		case COMMAND::TURN_LED_OFF: led = OFF; break;
		case COMMAND::TURN_LED_ON:  led = ON;  break;
	}
}
