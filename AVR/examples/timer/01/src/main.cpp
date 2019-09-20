#include <avr/io.h>

#include <util.h>
#include <adc.h>
#include <led.h>
#include <timer/timer0.h>

static Led<PORT::B, 5> led;

static uint16_t counter_top = 0;

void init(void)
{
	Timer0::Init timer0_init;
	timer0_init.mode                      = Timer0::MODE::CTC;
	timer0_init.clock_source              = Timer0::CLOCK_SOURCE::IO_CLK_OVER_64;
	timer0_init.on_output_compare_match_A = []()
	{
		static uint16_t counter = 0;
		if (counter >= counter_top)
		{
			led.toggle();
			counter = 0;
		}
		counter++;
	};
	timer0.init(timer0_init);

	adc.init({ });
	adc.channel(5);
	adc.on_conversion = []() { counter_top = adc.value; };

	sei();

	timer0.enable_output_compare_match_A_interrupt();
	adc.start_sampling();
}

int main(void)
{
	init();

	while (1) { }
}
