#include <i2c/i2c.h>
#include <usart/usart0.h>

#define SDA_PORT PORT::C
#define SCL_PORT PORT::C
#define SDA_PIN  1
#define SCL_PIN  0
#include <avr/interrupt.h>

#define SLAVE_ADDRESS 0b0100111
// #define SLAVE_ADDRESS 0b1010010

I2c<SDA_PORT, SDA_PIN, SCL_PORT, SCL_PIN, 30> i2c;

void init(void)
{
	usart0.init({ TIO_BAUD });
	stdout = usart0.stream();

	sei();
}

int main(void)
{
	init();

	while (true)
	{
		bool ack_addr = NACK;
		bool ack_data = NACK;
		uint8_t  data = 0x00;

		i2c.start();
		ack_addr = i2c.write((SLAVE_ADDRESS << 1) | 0);
		ack_data = i2c.write(0xFF);
		i2c.stop();
		if (ack_addr) printf("ADDR: ACK \n");
		else          printf("ADDR: NACK\n");
		if (ack_data) printf("DATA: ACK \n");
		else          printf("DATA: NACK\n");

		i2c.start();
		ack_addr = i2c.write((SLAVE_ADDRESS << 1) | 1);
		data     = i2c.read();
		i2c.stop();
		if (ack_addr) printf("ADDR: ACK \n");
		else          printf("ADDR: NACK\n");
		printf("DATA: 0x%02X\n", data);

		putchar('\n');
	}
}
