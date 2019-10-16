#include <gpio.h>
#include <usart/usart0.h>

using namespace Usart;

Gpio<PORT::D, 7> pin(OUTPUT);

void init(void)
{
	usart0.init({ TIO_BAUD });
	stdout = usart0.output_stream();
}

int main(void)
{
	init();

	while (true)
	{
	}
}
