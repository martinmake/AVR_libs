#include <standard/standard.h>
#include <usart/usart.h>
#include <eeprom/eeprom.h>

void init(void)
{
	Usart::begin(BAUD, F_CPU);
}

int main(void)
{
	init();

	for (uint16_t i = 0; i < 0x0200; i++)
		Eeprom::write(i, (i % 16) + ((i % 16) * 16));

	for (uint16_t i = 0; i < 0x0200; i++)
		Usart::sendf(20, "0x%04X: 0x%02X\n", i, Eeprom::read(i));

	while (1) {
	}
}
