#include <avr/io.h>

#define UTIL_DEFINE_SLEEP
#include <util.h>
#include <adc.h>
#include <led.h>
#include <usart/usart0.h>
#include <timer/timer0.h>

static Led<PORT::B, 5> led;

static uint16_t counter_top = 0;

void timer0_on_output_compare_match_A(void)
{
	static uint16_t counter = 0;
	if (counter >= counter_top)
	{
		led.toggle();
		counter = 0;
	}
	counter++;
}

void init(void)
{
	Timer0::Init timer0_init;
	timer0_init.clock_source              = Timer0::CLOCK_SOURCE::IO_CLK_OVER_64;
	timer0_init.ctc                       = true;
	timer0_init.on_output_compare_match_A = timer0_on_output_compare_match_A;
	timer0.init(timer0_init);

	adc.init({ });
	adc.channel(5);
	adc.on_conversion = [](uint16_t result) { counter_top = result; };

	sei();

	timer0.enable_output_compare_match_A_interrupt();
	adc.start_sampling();
}

int main(void)
{
	init();

	while (1) { }
}
