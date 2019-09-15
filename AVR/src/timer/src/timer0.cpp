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
	switch (init_struct.on_compare_match_output_A)
	{
		case ON_COMPARE_MATCH_OUTPUT::PASS:
			CLEAR(TCCR0A, COM0A0);
			CLEAR(TCCR0A, COM0A1);
			break;
		case ON_COMPARE_MATCH_OUTPUT::TOGGLE:
			SET  (TCCR0A, COM0A0);
			CLEAR(TCCR0A, COM0A1);
			SET(DDRD, PD6);
			break;
		case ON_COMPARE_MATCH_OUTPUT::CLEAR:
			CLEAR(TCCR0A, COM0A0);
			SET  (TCCR0A, COM0A1);
			SET(DDRD, PD6);
			break;
		case ON_COMPARE_MATCH_OUTPUT::SET:
			SET  (TCCR0A, COM0A0);
			SET  (TCCR0A, COM0A1);
			SET(DDRD, PD6);
			break;
	}
	switch (init_struct.on_compare_match_output_B)
	{
		case ON_COMPARE_MATCH_OUTPUT::PASS:
			CLEAR(TCCR0A, COM0B0);
			CLEAR(TCCR0A, COM0B1);
			break;
		case ON_COMPARE_MATCH_OUTPUT::TOGGLE:
			SET  (TCCR0A, COM0B0);
			CLEAR(TCCR0A, COM0B1);
			SET(DDRD, PD5);
			break;
		case ON_COMPARE_MATCH_OUTPUT::CLEAR:
			CLEAR(TCCR0A, COM0B0);
			SET  (TCCR0A, COM0B1);
			SET(DDRD, PD5);
			break;
		case ON_COMPARE_MATCH_OUTPUT::SET:
			SET  (TCCR0A, COM0B0);
			SET  (TCCR0A, COM0B1);
			SET(DDRD, PD5);
			break;
	}

	switch (init_struct.mode)
	{
		case MODE::NON_PWM:
			if (init_struct.ctc)
			{
				CLEAR(TCCR0A, WGM00);
				CLEAR(TCCR0A, WGM01);
				CLEAR(TCCR0A, WGM02);
			}
			else
			{
				CLEAR(TCCR0A, WGM00);
				SET  (TCCR0A, WGM01);
				CLEAR(TCCR0A, WGM02);
			}
			break;
		case MODE::FAST_PWM:
			SET  (TCCR0A, WGM00);
			SET  (TCCR0A, WGM01);
			SET  (TCCR0A, WGM02);
			break;
		case MODE::PHASE_CORRECT_PWM:
			SET  (TCCR0A, WGM00);
			CLEAR(TCCR0A, WGM01);
			SET  (TCCR0A, WGM02);
			break;
	}

	switch (init_struct.clock_source)
	{
		case CLOCK_SOURCE::IO_CLK_OVER_1:
			SET  (TCCR0B, CS00);
			CLEAR(TCCR0B, CS01);
			CLEAR(TCCR0B, CS02);
			break;
		case CLOCK_SOURCE::IO_CLK_OVER_8:
			SET  (TCCR0B, CS00);
			CLEAR(TCCR0B, CS01);
			CLEAR(TCCR0B, CS02);
			break;
		case CLOCK_SOURCE::IO_CLK_OVER_64:
			SET  (TCCR0B, CS00);
			SET  (TCCR0B, CS01);
			CLEAR(TCCR0B, CS02);
			break;
		case CLOCK_SOURCE::IO_CLK_OVER_256:
			CLEAR(TCCR0B, CS00);
			CLEAR(TCCR0B, CS01);
			SET  (TCCR0B, CS02);
			break;
		case CLOCK_SOURCE::IO_CLK_OVER_1024:
			SET  (TCCR0B, CS00);
			CLEAR(TCCR0B, CS01);
			SET  (TCCR0B, CS02);
			break;
		case CLOCK_SOURCE::EXTERNAL_ON_FALLING_EDGE:
			CLEAR(TCCR0B, CS00);
			SET  (TCCR0B, CS01);
			SET  (TCCR0B, CS02);
			break;
		case CLOCK_SOURCE::EXTERNAL_ON_RISING_EDGE:
			SET  (TCCR0B, CS00);
			SET  (TCCR0B, CS01);
			SET  (TCCR0B, CS02);
			break;
	}

	output_compare_register_A(init_struct.output_compare_value_A);
	output_compare_register_B(init_struct.output_compare_value_B);

	on_output_compare_match_A = init_struct.on_output_compare_match_A;
	on_output_compare_match_B = init_struct.on_output_compare_match_B;
	on_overflow               = init_struct.on_overflow;
}

ISR(TIMER0_OVF_vect) { timer0.on_overflow(); }

ISR(TIMER0_COMPA_vect) { timer0.on_output_compare_match_A(); }
ISR(TIMER0_COMPB_vect) { timer0.on_output_compare_match_B(); }

#endif
