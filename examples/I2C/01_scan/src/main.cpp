#include <avr/io.h>
#include <avr/interrupt.h>

#include <standard/standard.h>
#include <usart/usart.h>
#include <i2c/i2c.h>

#define F_SCL 9600

uint8_t value = 0x00;

void init(void)
{
	Usart::begin(BAUD , F_CPU);
	I2c  ::begin(F_SCL, F_CPU);

	sei();
}

int main(void)
{
	init();

	Usart::sends("Scan initiated...\n");

	for (uint8_t addr = 0x04; addr <= 0x77; addr++) {
		I2c::read(addr, 1, &value);
		while (!I2c::transceive_completed)
			;

		if (!I2c::transceive_failed)
			Usart::sendf(20, "FOUND: 0x%02X\n", addr);
	}

	for (uint8_t addr = 0x04; addr <= 0x77; addr++) {
		I2c::read(addr, 1, &value);
		while (!I2c::transceive_completed)
			;

		if (!I2c::transceive_failed)
			Usart::sendc('!');
		else
			Usart::sendc('.');
	}
	Usart::sendc('\n');

	Usart::sends("Scan completed...\n");

	while (1) {
	}
}
