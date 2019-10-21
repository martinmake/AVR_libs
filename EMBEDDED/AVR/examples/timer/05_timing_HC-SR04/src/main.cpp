#include <util.h>
#include <gpio.h>
#include <timer/timer0.h>
#include <usart/usart0.h>
#include <external_interrupt.h>

using namespace Timer;
using namespace Usart;

static bool pulse_end;

static Gpio<PORT::D, 7> trig(OUTPUT);
static Gpio<PORT::D, 6> echo(INPUT);

void init(void)
{
	usart0.init({ TIO_BAUD });
	stdout = usart0.stream();

	Timer0::Spec spec;
	spec.mode         = Timer0::MODE::NORMAL;
	spec.clock_source = Timer0::CLOCK_SOURCE::IO_CLK_OVER_8;
	timer0.init(spec);

	ExternalInterrupt edge_detector
	({
		ExternalInterrupt::INTERRUPT0,
		ExternalInterrupt::SENSE::FALLING_EDGE,
		[]() { pulse_end = true; }
	});

	sei();
}

int main(void)
{
	init();

	while (true)
	{
		timer0.unpause();
		timer0.count(0);
		trig = HIGH;
		while (timer0.count() < 20) {}
		trig = LOW;

		while (!echo) {}

		pulse_end = false;
		bool timed_out = true;
		timer0.count(0);
		while (timer0.count() < 200)
		{
			if (pulse_end)
			{
				timer0.pause();
				timed_out = false;
				break;
			}
		}

		if (timed_out) puts("TIMED OUT!");
		else
		{
			float distance = timer0.count() / 2; // TODO: COMPUTE DISTANCE
			printf("DISTANCE: %fm\n", distance);
		}
	}
}
