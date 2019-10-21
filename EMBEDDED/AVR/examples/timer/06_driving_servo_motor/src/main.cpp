#include <util.h>
#include <gpio.h>
#include <timer/timer1.h>
#include <usart/usart0.h>

#define PRESCALER (64)
#define PERIOD    (0.02)
#define F_TIM     (F_CPU / PRESCALER)
#define TOP_VALUE (uint16_t (F_TIM * PERIOD - 1))

using namespace Timer;
using namespace Usart;

Gpio<PORT::B, 1> servo1(OUTPUT);
Gpio<PORT::B, 2> servo2(OUTPUT);

void init(void)
{
	usart0.init({ TIO_BAUD });
	stdout = usart0.stream();
	stderr = usart0.stream();
	stdin  = usart0.stream();

	Timer1::Spec spec;
	spec.mode                                 = Timer1::MODE::FAST_PWM;
	spec.top                                  = Timer1::TOP::INPUT_CAPTURE_VALUE;
	spec.pin_action_on_output_compare_match_A = Timer1::PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::CLEAR;
	spec.pin_action_on_output_compare_match_B = Timer1::PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::CLEAR;
	spec.clock_source                         = Timer1::CLOCK_SOURCE::IO_CLK_OVER_64;
	timer1.init(spec);
	timer1.input_capture_value(TOP_VALUE);

	sei();
}

int main(void)
{
	init();

	while (true)
	{
		timer1.output_compare_value_A(535);
		timer1.output_compare_value_B(535);
	}
}
