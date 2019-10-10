#include <system_clock.h>
#include <vl53l0x/software.h>
#include <usart/usart0.h>

#define SDA_PORT PORT::B
#define SCL_PORT PORT::B
#define SDA_PIN  0
#define SCL_PIN  1

using namespace Usart;

Vl53l0x::Software<SDA_PORT, SDA_PIN, SCL_PORT, SCL_PIN> vl53l0x;

void init(void)
{
	system_clock.init({ SystemClock::TIMER::TIMER0 });

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
		printf("[0xC0] GOT:   0x%02X, EXPECTED: 0xEE\n", vl53l0x.read_register_8bit (0xC0));
		printf("[0xC1] GOT:   0x%02X, EXPECTED: 0xAA\n", vl53l0x.read_register_8bit (0xC1));
		printf("[0xC2] GOT:   0x%02X, EXPECTED: 0x10\n", vl53l0x.read_register_8bit (0xC2));
		printf("[0x51] GOT: 0x%04X, EXPECTED: 0x0099\n", vl53l0x.read_register_16bit(0x51));
		printf("[0x61] GOT: 0x%04X, EXPECTED: 0x0000\n", vl53l0x.read_register_16bit(0x61));
		putchar('\n');
		system_clock.sleep(500);
	}
}
