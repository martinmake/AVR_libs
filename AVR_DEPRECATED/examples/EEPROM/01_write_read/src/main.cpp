#include <standard/standard.h>
#include <usart/usart0.h>
#include <eeprom/eeprom.h>

Usart0 usart0(TIO_BAUD, F_CPU);
Eeprom eeprom;

void init(void)
{
	sei();
}

int main(void)
{
	init();

	for (uint16_t i = 0; i < 0x0200; i++) {
		eeprom[i] << i;
	}

	for (uint16_t i = 0; i < 0x0200; i++) {
		uint8_t data;
		eeprom[i] >> data;
		usart0.sendf(20, "0x%04X: 0x%02X\n", i, data);
	}
}
