#if defined(__AVR_ATmega328P__)

#include <avr/interrupt.h>

#include <util.h>

#include "timer/timer0.h"

namespace Timer
{
	Timer0 timer0;

	// CONSTRUCTORS
	Timer0::Timer0(const Spec& spec)
	{
		init(spec);
	}

	// METHODS
	void Timer0::init(const Spec& spec)
	{
		pin_action_on_output_compare_match_A(spec.pin_action_on_output_compare_match_A);
		pin_action_on_output_compare_match_B(spec.pin_action_on_output_compare_match_B);

		mode                                (spec.mode                                );
		clock_source                        (spec.clock_source                        );

		output_compare_register_A(spec.output_compare_value_A);
		output_compare_register_B(spec.output_compare_value_B);

		on_output_compare_match_A(spec.on_output_compare_match_A);
		on_output_compare_match_B(spec.on_output_compare_match_B);
		on_overflow              (spec.on_overflow              );
	}
}

ISR(TIMER0_OVF_vect  ) { Timer::timer0.call_on_overflow              (); }
ISR(TIMER0_COMPA_vect) { Timer::timer0.call_on_output_compare_match_A(); }
ISR(TIMER0_COMPB_vect) { Timer::timer0.call_on_output_compare_match_B(); }

#endif
