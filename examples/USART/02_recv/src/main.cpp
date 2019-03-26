#include <avr/io.h>
#include <util/delay.h>

#include <standard/standard.h>
#include <usart/usart.h>

#define LED_OFF '0'
#define LED_ON  '1'

Pin led({PORTB, PB5});

void init(void)
{
// 	Usart::INIT usart_init;
// 	usart_init.tx              = Usart::RX::ON;
// 	usart_init.character_size  = Usart::CHARACTER_SIZE::S8;
// 	usart_init.stop_bit_select = Usart::STOP_BIT_SELECT::S2;
// 	usart_init.baud            = BAUD;
// 	usart_init.f_osc           = F_CPU;
// 	Usart::begin(&usart_init);

	Usart::begin(BAUD, F_CPU);
	led.dd.set();
}

int main(void)
{
	init();

	while (1) {
		char command = Usart::recvc();

		switch (command) {
			case LED_OFF: led.port.clear(); break;
			case LED_ON:  led.port.set()  ; break;
		}
	}
}
