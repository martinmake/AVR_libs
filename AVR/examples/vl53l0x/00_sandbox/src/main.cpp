#include <vl53l0x/software.h>
#include <usart/usart0.h>

#define SDA_PORT PORT::B
#define SCL_PORT PORT::B
#define SDA_PIN  0
#define SCL_PIN  1

Vl53l0x::Software<SDA_PORT, SDA_PIN, SCL_PORT, SCL_PIN> vl53l0x;

void init(void)
{
	usart0.init({ TIO_BAUD });
	stdout = usart0.stream();

	sei();

	while (true)
	{
		vl53l0x.start();
		vl53l0x.write(vl53l0x.address() << 1);
		vl53l0x.stop();

		if (vl53l0x.ack())
		{
			printf("DEVICE[0x%02X] FOUND!\n", vl53l0x.address());
			break;
		}
		else printf("DEVICE[0x%02X] NOT FOUND!\n", vl53l0x.address());
		_delay_ms(10);
	}
}

int main(void)
{
	init();

	while (true)
	{
		printf("0x%02X\n", vl53l0x.read_register_8bit(0xC0));
		_delay_ms(10);
	}
}
