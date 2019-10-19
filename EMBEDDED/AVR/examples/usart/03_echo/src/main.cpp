#include <usart/usart0.h>

using namespace Usart;

void init(void)
{
	usart0.init({ TIO_BAUD });
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
