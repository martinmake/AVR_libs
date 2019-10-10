#include <avr/io.h>
#include <avr/interrupt.h>

#include <inttypes.h>
#include <string.h>

#include <util.h>
#include <led.h>
#include <usart/usart0.h>

using namespace Usart;

void init(void)
{
	Usart0::Init usart0_init;
	usart0_init.baud = TIO_BAUD;
	usart0 = Usart0(usart0_init);
}

int main(void)
{
	init();

	while (1)
	{
		char c;
		usart0 >> c;
		usart0 << c;
	}
}
