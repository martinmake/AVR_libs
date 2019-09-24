#include <i2c/i2c.h>
#include <usart/usart0.h>

#define SDA_PORT PORT::C
#define SCL_PORT PORT::C
#define SDA_PIN  1
#define SCL_PIN  0

// #define DEV_ADDRESS 0b0100111
#define DEV_ADDRESS 0b0101001

I2c<SDA_PORT, SDA_PIN, SCL_PORT, SCL_PIN, 5> i2c;

void init(void)
{
	usart0.init({ TIO_BAUD });
	stdout = usart0.stream();

	sei();

	while (true)
	{
		bool ack = NACK;

		i2c.start();
		ack = i2c.write((DEV_ADDRESS << 1));
		i2c.stop();

		if (ack) break;
		printf("DEVICE[0x%02X] NOT FOUND!\n", DEV_ADDRESS);
	}
}

int main(void)
{
	init();

	while (true)
	{
		uint8_t data[3] = { 0 };
		i2c.read_reg_8bit(DEV_ADDRESS, 0xC0, data, 3);
		printf("[0xC0] 0x%02X\n", data[0]);
		printf("[0xC1] 0x%02X\n", data[1]);
		printf("[0xC2] 0x%02X\n", data[2]);
		printf("[0xC0] 0x%02X\n", i2c.read_reg_8bit (DEV_ADDRESS, 0xC0));
		printf("[0xC1] 0x%02X\n", i2c.read_reg_8bit (DEV_ADDRESS, 0xC1));
		printf("[0xC2] 0x%02X\n", i2c.read_reg_8bit (DEV_ADDRESS, 0xC2));
		printf("[0x51] 0x%04X\n", i2c.read_reg_16bit(DEV_ADDRESS, 0x51));
		printf("[0x61] 0x%04X\n", i2c.read_reg_16bit(DEV_ADDRESS, 0x61));
		putchar('\n');
		_delay_ms(10);
	}
}
