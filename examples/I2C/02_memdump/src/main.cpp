#include <avr/io.h>

#include <standard/standard.h>
#include <usart/usart.h>
#include <i2c/i2c.h>

#define F_SCL 9600

void init(void)
{
	Usart::begin(BAUD , F_CPU);
	I2c  ::begin(F_SCL, F_CPU);
}

int main(void)
{
	init();

	Usart::sends("Memory dump initiated...\n");

	for (uint8_t i = 0x04; i <= 0x77; i++) {
		uint8_t val = I2c::read(i);
		Usart::sendf(20, "0x%02X: 0x%02X\n", i, val);
	}

	Usart::sends("Memory dump completed...\n");

	while (1) {
	}
}
