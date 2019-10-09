#include <util.h>
#include <led.h>
#include <adc.h>
#include <timer/timer1.h>

using namespace Timer;

static Led<PORT::D, 5> led;

void init(void)
{
	Timer1::Init timer1_init;
	timer1_init.mode                      = Timer1::MODE::FAST_PWM;
	timer1_init.clock_source              = Timer1::CLOCK_SOURCE::IO_CLK_OVER_64;
	timer1_init.on_output_compare_match_A = []() { led.toggle(); };
	timer1.init(timer1_init);

	adc.init({ });
	adc.channel(5);
	adc.on_conversion = []() { timer1.output_compare_register_A(adc.value); };

	sei();

	adc.start_sampling();
}

int main(void)
{
	init();

	while (1) { }
}
