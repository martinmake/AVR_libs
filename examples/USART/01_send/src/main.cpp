#include <avr/io.h>
#include <util/delay.h>

#include <usart/usart.h>

void init(void)
{
// 	Usart::INIT usart_init;
// 	usart_init.tx              = Usart::TX::ON;
// 	usart_init.character_size  = Usart::CHARACTER_SIZE::S8;
// 	usart_init.stop_bit_select = Usart::STOP_BIT_SELECT::S2;
// 	usart_init.baud            = BAUD;
// 	usart_init.f_osc           = F_CPU;
// 	Usart::begin(&usart_init);

	Usart::begin(BAUD, F_CPU);
}

int main(void)
{
	init();

	while (1) {
		Usart::sendc('X');
		Usart::sendc('\n');
		Usart::sends("TEST!");
		Usart::sendc('\n');
		Usart::sendf(10, "2 + 2 = %d", 2 + 2);
		Usart::sendc('\n');
		_delay_ms(1000);
	}
}
