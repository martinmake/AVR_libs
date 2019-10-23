#if defined(__AVR_ATmega328P__)

#include "timer/timer1.h"

namespace Timer
{
	Timer1 timer1;

	// CONSTRUCTORS
	Timer1::Timer1(const Spec& spec)
	{
		init(spec);
	}

	// METHODS
	void Timer1::init(const Spec& spec)
	{
		mode(spec.mode);
		top (spec.top );

		input_capture(spec.input_capture);

		pin_action_on_output_compare_match_A(spec.pin_action_on_output_compare_match_A);
		pin_action_on_output_compare_match_B(spec.pin_action_on_output_compare_match_B);

		clock_source(spec.clock_source);

		on<INTERRUPT::ON_OUTPUT_COMPARE_MATCH_A>(spec.on_output_compare_match_A);
		on<INTERRUPT::ON_OUTPUT_COMPARE_MATCH_B>(spec.on_output_compare_match_B);
		on<INTERRUPT::ON_INPUT_CAPTURE         >(spec.on_input_capture         );
		on<INTERRUPT::ON_OVERFLOW              >(spec.on_overflow              );
	}
}

ISR(TIMER1_COMPA_vect) { Timer::timer1.call<Timer::Timer1::INTERRUPT::ON_OUTPUT_COMPARE_MATCH_A>(); }
ISR(TIMER1_COMPB_vect) { Timer::timer1.call<Timer::Timer1::INTERRUPT::ON_OUTPUT_COMPARE_MATCH_B>(); }
ISR(TIMER1_CAPT_vect ) { Timer::timer1.call<Timer::Timer1::INTERRUPT::ON_INPUT_CAPTURE         >(); }
ISR(TIMER1_OVF_vect  ) { Timer::timer1.call<Timer::Timer1::INTERRUPT::ON_OVERFLOW              >(); }

#endif
