#if defined(__AVR_ATmega328P__)

#include <avr/interrupt.h>

#include <util.h>

#include "timer/timer0.h"

namespace Timer
{
	Timer0 timer0;

	Timer0::Timer0(void)
	{
	}
	Timer0::Timer0(const Spec& Spec)
	{
		init(Spec);
	}
	void Timer0::init(const Spec& Spec)
	{
		on_compare_match_output_A_pin_action(Spec.on_compare_match_output_A_pin_action);
		on_compare_match_output_B_pin_action(Spec.on_compare_match_output_B_pin_action);
		mode                                (Spec.mode                                );
		clock_source                        (Spec.clock_source                        );

		output_compare_register_A(Spec.output_compare_value_A);
		output_compare_register_B(Spec.output_compare_value_B);

		on_output_compare_match_A(Spec.on_output_compare_match_A);
		on_output_compare_match_B(Spec.on_output_compare_match_B);
		on_overflow              (Spec.on_overflow              );
	}
}

ISR(TIMER0_OVF_vect  ) { Timer::timer0.call_on_overflow              (); }
ISR(TIMER0_COMPA_vect) { Timer::timer0.call_on_output_compare_match_A(); }
ISR(TIMER0_COMPB_vect) { Timer::timer0.call_on_output_compare_match_B(); }

#endif
