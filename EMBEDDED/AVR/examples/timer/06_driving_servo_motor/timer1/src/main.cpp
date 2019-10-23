#include <util.h>
#include <core/gpio.h>
#include <core/timer/timer1.h>
// #include <core/usart/usart0.h>
// #include <system_clock.h>

#define PRESCALER 8
#define PERIOD    0.02

#define F_TIM     (F_CPU / PRESCALER)
#define TOP_VALUE (uint16_t (F_TIM * PERIOD - 1))

#define OUTPUT_COMPARE_VALUE_CLOCKWISE      (uint16_t (F_CPU / PRESCALER / 1000000 * 1000))
#define OUTPUT_COMPARE_VALUE_STOP           (uint16_t (F_CPU / PRESCALER / 1000000 * 1500))
#define OUTPUT_COMPARE_VALUE_ANTI_CLOCKWISE (uint16_t (F_CPU / PRESCALER / 1000000 * 2000))

using namespace Core;

Gpio<GPIO::PORT::B, 1> servo1(GPIO::MODE::OUTPUT);
Gpio<GPIO::PORT::B, 2> servo2(GPIO::MODE::OUTPUT);

void initialize(void)
{
	// system_clock.initialize({ SystemClock::TIMER0 });

	// usart0.initialize({ TIO_BAUD });
	// stdout = usart0.stream();
	// stderr = usart0.stream();
	// stdin  = usart0.stream();

	{
		using namespace Timer;
		using namespace TIMER;

		Spec spec;
		spec.mode                                 = MODE::FAST_PWM;
		spec.top                                  = TOP::INPUT_CAPTURE_VALUE;
		spec.pin_action_on_output_compare_match_A = PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::CLEAR;
		spec.pin_action_on_output_compare_match_B = PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::CLEAR;
		spec.clock_source                         = CLOCK_SOURCE::IO_CLK_OVER_8;
		timer1.initialize(spec);
		timer1.input_capture_value(TOP_VALUE);
	}

	sei();
}

int main(void)
{
	initialize();

	servo1 = ON;
	servo2 = ON;

	while (true)
	{
		timer1.output_compare_value_A(OUTPUT_COMPARE_VALUE_CLOCKWISE);
		timer1.output_compare_value_B(OUTPUT_COMPARE_VALUE_CLOCKWISE);
		_delay_ms(1000);
		// system_clock.sleep(1000);

		timer1.output_compare_value_A(OUTPUT_COMPARE_VALUE_STOP);
		timer1.output_compare_value_B(OUTPUT_COMPARE_VALUE_STOP);
		_delay_ms(1000);
		// system_clock.sleep(1000);

		timer1.output_compare_value_A(OUTPUT_COMPARE_VALUE_ANTI_CLOCKWISE);
		timer1.output_compare_value_B(OUTPUT_COMPARE_VALUE_ANTI_CLOCKWISE);
		_delay_ms(1000);
		// system_clock.sleep(1000);

		timer1.output_compare_value_A(OUTPUT_COMPARE_VALUE_STOP);
		timer1.output_compare_value_B(OUTPUT_COMPARE_VALUE_STOP);
		_delay_ms(1000);
		// system_clock.sleep(1000);
	}
}
