#if defined(__AVR_ATmega328P__)

#include "timer/timer2.h"

namespace Timer
{
	Timer2 timer2;

	Timer2::Timer2(void)
	{
	}
	Timer2::Timer2(const Init& init_struct)
	{
		init(init_struct);
	}
	void Timer2::init(const Init& init_struct)
	{
		switch (init_struct.on_compare_match_output_A)
		{
			case ON_COMPARE_MATCH_OUTPUT::PASS:
				CLEAR(TCCR2A, COM2A0);
				CLEAR(TCCR2A, COM2A1);
				break;
			case ON_COMPARE_MATCH_OUTPUT::TOGGLE:
				SET  (TCCR2A, COM2A0);
				CLEAR(TCCR2A, COM2A1);
				SET(DDRB, PB3);
				break;
			case ON_COMPARE_MATCH_OUTPUT::CLEAR:
				CLEAR(TCCR2A, COM2A0);
				SET  (TCCR2A, COM2A1);
				SET(DDRB, PB3);
				break;
			case ON_COMPARE_MATCH_OUTPUT::SET:
				SET  (TCCR2A, COM2A0);
				SET  (TCCR2A, COM2A1);
				SET(DDRB, PB3);
				break;
		}
		switch (init_struct.on_compare_match_output_B)
		{
			case ON_COMPARE_MATCH_OUTPUT::PASS:
				CLEAR(TCCR2A, COM2B0);
				CLEAR(TCCR2A, COM2B1);
				break;
			case ON_COMPARE_MATCH_OUTPUT::TOGGLE:
				SET  (TCCR2A, COM2B0);
				CLEAR(TCCR2A, COM2B1);
				SET(DDRD, PD3);
				break;
			case ON_COMPARE_MATCH_OUTPUT::CLEAR:
				CLEAR(TCCR2A, COM2B0);
				SET  (TCCR2A, COM2B1);
				SET(DDRD, PD3);
				break;
			case ON_COMPARE_MATCH_OUTPUT::SET:
				SET  (TCCR2A, COM2B0);
				SET  (TCCR2A, COM2B1);
				SET(DDRD, PD3);
				break;
		}

		switch (init_struct.mode)
		{
			case MODE::NORMAL:
				CLEAR(TCCR2A, WGM20);
				CLEAR(TCCR2A, WGM21);
				CLEAR(TCCR2A, WGM22);
				break;
			case MODE::CTC:
				CLEAR(TCCR2A, WGM20);
				SET  (TCCR2A, WGM21);
				CLEAR(TCCR2A, WGM22);
				break;
			case MODE::FAST_PWM:
				SET  (TCCR2A, WGM20);
				SET  (TCCR2A, WGM21);
				SET  (TCCR2A, WGM22);
				break;
			case MODE::PHASE_CORRECT_PWM:
				SET  (TCCR2A, WGM20);
				CLEAR(TCCR2A, WGM21);
				SET  (TCCR2A, WGM22);
				break;
		}

		switch (init_struct.clock_source)
		{
			case CLOCK_SOURCE::IO_CLK_OVER_1:
				SET  (TCCR2B, CS20);
				CLEAR(TCCR2B, CS21);
				CLEAR(TCCR2B, CS22);
				break;
			case CLOCK_SOURCE::IO_CLK_OVER_8:
				SET  (TCCR2B, CS20);
				CLEAR(TCCR2B, CS21);
				CLEAR(TCCR2B, CS22);
				break;
			case CLOCK_SOURCE::IO_CLK_OVER_64:
				SET  (TCCR2B, CS20);
				SET  (TCCR2B, CS21);
				CLEAR(TCCR2B, CS22);
				break;
			case CLOCK_SOURCE::IO_CLK_OVER_256:
				CLEAR(TCCR2B, CS20);
				CLEAR(TCCR2B, CS21);
				SET  (TCCR2B, CS22);
				break;
			case CLOCK_SOURCE::IO_CLK_OVER_1024:
				SET  (TCCR2B, CS20);
				CLEAR(TCCR2B, CS21);
				SET  (TCCR2B, CS22);
				break;
			case CLOCK_SOURCE::EXTERNAL_ON_FALLING_EDGE:
				CLEAR(TCCR2B, CS20);
				SET  (TCCR2B, CS21);
				SET  (TCCR2B, CS22);
				break;
			case CLOCK_SOURCE::EXTERNAL_ON_RISING_EDGE:
				SET  (TCCR2B, CS20);
				SET  (TCCR2B, CS21);
				SET  (TCCR2B, CS22);
				break;
		}

		output_compare_register_A(init_struct.output_compare_value_A);
		output_compare_register_B(init_struct.output_compare_value_B);

		on_output_compare_match_A = init_struct.on_output_compare_match_A;
		on_output_compare_match_B = init_struct.on_output_compare_match_B;
		on_overflow               = init_struct.on_overflow;

		if (on_output_compare_match_A) enable_output_compare_match_A_interrupt();
		if (on_output_compare_match_B) enable_output_compare_match_B_interrupt();
		if (on_overflow              ) enable_overflow_interrupt();
	}
}

ISR(TIMER0_OVF_vect) { Timer::timer2.on_overflow(); }

ISR(TIMER0_COMPA_vect) { Timer::timer2.on_output_compare_match_A(); }
ISR(TIMER0_COMPB_vect) { Timer::timer2.on_output_compare_match_B(); }

#endif
