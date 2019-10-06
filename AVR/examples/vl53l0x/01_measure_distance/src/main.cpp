#include <vl53l0x/software.h>
#include <system_clock.h>
#include <usart/usart0.h>

#define SDA_PORT PORT::C
#define SCL_PORT PORT::C
#define SDA_PIN  4
#define SCL_PIN  5

using namespace Usart;

Vl53l0x::Software<SDA_PORT, SDA_PIN, SCL_PORT, SCL_PIN> vl53l0x;

void init(void)
{
	system_clock.init({ SystemClock::TIMER::TIMER0 });

	usart0.init({ TIO_BAUD });
	stdout = usart0.stream();

	sei();

	vl53l0x.init();
	while (!vl53l0x.is_connected())
	{
		printf("DEVICE[0x%02X] NOT FOUND!\n", vl53l0x.address());
		system_clock.sleep(10);
	}
	printf("DEVICE[0x%02X] FOUND!\n", vl53l0x.address());
}

int main(void)
{
	init();

	while (true)
		printf("%u\n", vl53l0x.range());
}
