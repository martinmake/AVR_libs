#include <util.h>
#include <gpio.h>
#include <usart/usart0.h>

Gpio<PORT::D, 7, INPUT> input_pin;

void init(void)
{
	usart0.init({ TIO_BAUD });
	stdout = usart0.stream();
}

int main(void)
{
	init();

	while (true)
	{
		if (input_pin) printf("HIGH\n");
		else           printf("LOW \n");
		// CLEAR(DDRD,  7);
		// CLEAR(PORTD, 7);
		// if (IS_SET(PIND, 7)) printf("HIGH\n");
		// else                 printf("LOW \n");
	}
}
