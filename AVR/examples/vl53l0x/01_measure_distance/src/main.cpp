#include <usart/usart0.h>
#include <i2c/software.h>

#define SDA_PORT PORT::B
#define SCL_PORT PORT::B
#define SDA_PIN  0
#define SCL_PIN  1

I2c::Software<SDA_PORT, SDA_PIN, SCL_PORT, SCL_PIN, 5> i2c;

void init(void)
{
	usart0.init({ TIO_BAUD });
	stdout = usart0.stream();

	sei();
}

int main(void)
{
	init();

	usart0 << "[*] SCAN INITIATED!\n";

	bool at_least_one_device = false;
	for (uint8_t addr = 0x04; addr <= 0x77; addr++) {
		i2c.start();
		i2c.write(addr << 1);
		i2c.stop();

		if (i2c.ack())
		{
			printf("[+] FOUND: 0x%02X\n", addr);
			at_least_one_device = true;
		}
	}

	for (uint8_t addr = 0x04; addr <= 0x77; addr++) {
		i2c.start();
		i2c.write(addr << 1);
		i2c.stop();

		if (i2c.ack()) putchar('!');
		else           putchar('.');
	}
	putchar('\n');

	usart0 << "[*] SCAN COMPLETED!\n";
	if (!at_least_one_device)
		usart0 << "[-] NO DEVICES WERE FOUND!\n";

	while (true) { }
}
