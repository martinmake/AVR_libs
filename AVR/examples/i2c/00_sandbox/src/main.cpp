#include <i2c/i2c.h>
#include <usart/usart0.h>

#define SDA_PORT PORT::D
#define SCL_PORT PORT::D
#define SDA_PIN  3
#define SCL_PIN  2

// #define SLAVE_ADDRESS 0b0100000
#define SLAVE_ADDRESS 0b1010010

void init(void)
{
	usart0.init({ TIO_BAUD });
	stdout = usart0.stream();

	sei();
}

int main(void)
{
	init();

	I2c<SDA_PORT, SDA_PIN, SCL_PORT, SCL_PIN, 500> i2c;

	while (true)
	{
		bool ack_addr = NACK;
		bool ack_data = NACK;
		uint8_t  data = 0x00;

		i2c.start();
		ack_addr = i2c.write(SLAVE_ADDRESS << 1);
		ack_data = i2c.write(0xAA);
		i2c.stop();

		i2c.start();
		ack_addr = i2c.write(SLAVE_ADDRESS << 1);
		data     = i2c.read();
		i2c.stop();

		if (ack_addr) printf("ADDR: ACK \n");
		else          printf("ADDR: NACK\n");
		if (ack_data) printf("DATA: ACK \n");
		else          printf("DATA: NACK\n");
		printf("DATA: 0x%02X\n", data);
		putchar('\n');
	}
}
