#include <avr/io.h>

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

	usart0 << "Memory dump initiated...\n";

	for (uint8_t i = 0x04; i <= 0x77; i++) {
		uint8_t val;
		i2c.read(i, 1, &val);
		usart0.sendf(20, "0x%02X: 0x%02X\n", i, val);
	}

	usart0 << "Memory dump completed...\n";

	while (1) {
	}
}
