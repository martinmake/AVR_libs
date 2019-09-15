#define UTIL_DEFINE_SLEEP
#include <util/util.h>
#include <pin/pin.h>

Pin<Port::B, 5> led;

inline void init(void)
{
}

int main(void)
{
	init();

	while (true)
	{
		led = HIGH;
		sleep(200);
		led = LOW;
		sleep(200);
	}
}
