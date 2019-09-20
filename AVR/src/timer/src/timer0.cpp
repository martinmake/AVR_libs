#if defined(__AVR_ATmega328P__)

#include <avr/interrupt.h>

#include <util.h>

#include "timer/timer0.h"

Timer0 timer0;

Timer0::Timer0(void)
{
}
Timer0::Timer0(const Init& init_struct)
{
	init(init_struct);
}
void Timer0::init(const Init& init_struct)
{
	on_compare_match_output_A_pin_action(init_struct.on_compare_match_output_A_pin_action);
	on_compare_match_output_B_pin_action(init_struct.on_compare_match_output_B_pin_action);
	mode                                (init_struct.mode                                );
	clock_source                        (init_struct.clock_source                        );

	output_compare_register_A(init_struct.output_compare_value_A);
	output_compare_register_B(init_struct.output_compare_value_B);

	on_output_compare_match_A(init_struct.on_output_compare_match_A);
	on_output_compare_match_B(init_struct.on_output_compare_match_B);
	on_overflow              (init_struct.on_overflow              );
}

ISR(TIMER0_OVF_vect  ) { timer0.call_on_overflow              (); }
ISR(TIMER0_COMPA_vect) { timer0.call_on_output_compare_match_A(); }
ISR(TIMER0_COMPB_vect) { timer0.call_on_output_compare_match_B(); }

#endif
