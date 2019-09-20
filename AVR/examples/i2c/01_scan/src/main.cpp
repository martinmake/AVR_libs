#include <avr/io.h>
#include <avr/interrupt.h>

#include <standard/standard.h>
#include <usart/usart0.h>
#include <i2c/i2c.h>

#define F_SCL 9600

Usart0 usart0(TIO_BAUD, F_CPU);
I2c    i2c   (F_SCL   , F_CPU);

void init(void)
{
	sei();
}

int main(void)
{
	init();

	uint8_t value = 0x00;

	usart0 << "Scan initiated...\n";

	for (uint8_t addr = 0x04; addr <= 0x77; addr++) {
		i2c.read(addr, 1, &value);
		i2c.wait_until_transceive_completed();

		if (!i2c.transceive_failed)
			usart0.sendf(20, "FOUND: 0x%02X\n", addr);
	}

	for (uint8_t addr = 0x04; addr <= 0x77; addr++) {
		i2c.read(addr, 1, &value);
		i2c.wait_until_transceive_completed();

		if (!i2c.transceive_failed)
			usart0 << '!';
		else
			usart0 << '.';
	}
	usart0 << '\n';

	usart0 << "Scan completed...\n";

	while (1) {
	}
}
