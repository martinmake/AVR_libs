#include <util.h>
#include <usart/usart0.h>

using namespace Usart;

void init(void)
{
	usart0.init({ TIO_BAUD });
	stdout = usart0.stream();

	sei();
}

int main(void)
{
	init();

	stty(FG_SET_COLOR, COLOR_NUMBER, 6);
	puts("stty(FG_SET_COLOR, COLOR_NUMBER, 5);");

	stty(RESET);
	puts("stty(RESET);");

	stty(FG_BLACK, BG_BLUE, BOLD, ITALIC);
	puts("stty(FG_BLACK, BG_BLUE, BOLD, ITALIC);");

	stty(FG_SET_COLOR, COLOR_RGB, 226, 36, 236);
	puts("stty(FG_SET_COLOR, COLOR_RGB, 226, 36, 236);");

	stty(NORMAL);
	puts("stty(NORMAL);");

	_delay_ms(1000);
}
