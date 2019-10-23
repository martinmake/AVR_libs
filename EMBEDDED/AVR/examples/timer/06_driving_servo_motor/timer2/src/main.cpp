#include <util.h>
#include <gpio.h>
#include <timer/timer2.h>
#include <usart/usart0.h>

using namespace Timer;
using namespace Usart;

Gpio<PORT::B, 1> servo1(OUTPUT);
Gpio<PORT::B, 2> servo2(OUTPUT);

void init(void)
{
	system_clock.init({ SystemClock::TIMER0 });

	usart0.init({ TIO_BAUD });
	stdout = usart0.stream();
	stderr = usart0.stream();
	stdin  = usart0.stream();

	Timer1::Spec spec;
	spec.mode                                 = Timer1::MODE::FAST_PWM;
	spec.top                                  = Timer1::TOP::INPUT_CAPTURE_VALUE;
	spec.pin_action_on_output_compare_match_A = Timer1::PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::CLEAR;
	spec.pin_action_on_output_compare_match_B = Timer1::PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::CLEAR;
	spec.clock_source                         = Timer1::CLOCK_SOURCE::IO_CLK_OVER_8;
	timer2.init(spec);
	timer2.input_capture_value(TOP_VALUE);

	sei();
}

int main(void)
{
	init();

	while (true)
	{
		timer2.unpause();
		timer2.count(0);
		trig = HIGH;
		while (timer0.count() < 20) {}
		trig = LOW;
	}
}
