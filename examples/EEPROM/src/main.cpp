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

	eeprom = 0;
	for (uint16_t i = 0; i < 0x0200; i++) {
		eeprom << i;
		eeprom++;
	}

	eeprom = 0;
	for (uint16_t i = 0; i < 0x0200; i++) {
		uint8_t data;
		eeprom >> data;
		eeprom++;
		usart0.sendf(20, "0x%04X: 0x%02X\n", i, data);
	}
}
