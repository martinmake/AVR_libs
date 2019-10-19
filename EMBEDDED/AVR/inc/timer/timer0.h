#ifndef _TIMER_TIMER0_H_
#define _TIMER_TIMER0_H_

#include <avr/io.h>

#include <util.h>

#include "timer/base.h"

namespace Timer
{
	class Timer0: virtual public Timer::Base
	{
		public: // TYPES
			enum class MODE : uint8_t
			{
				NORMAL,
				CTC,
				FAST_PWM,
				PHASE_CORRECT_PWM
			};
			enum class ON_COMPARE_MATCH_OUTPUT_PIN_ACTION : uint8_t
			{
				PASS,
				TOGGLE,
				CLEAR,
				SET
			};
			enum class CLOCK_SOURCE : uint8_t
			{
				IO_CLK_OVER_1,
				IO_CLK_OVER_8,
				IO_CLK_OVER_64,
				IO_CLK_OVER_256,
				IO_CLK_OVER_1024,
				EXTERNAL_ON_FALLING_EDGE,
				EXTERNAL_ON_RISING_EDGE
			};
			using on_output_compare_match_func = void (*)(void);
			using on_overflow_func             = void (*)(void);

			struct Spec
			{
				MODE                               mode                                 = MODE::NORMAL;
				ON_COMPARE_MATCH_OUTPUT_PIN_ACTION on_compare_match_output_A_pin_action = ON_COMPARE_MATCH_OUTPUT_PIN_ACTION::PASS;
				ON_COMPARE_MATCH_OUTPUT_PIN_ACTION on_compare_match_output_B_pin_action = ON_COMPARE_MATCH_OUTPUT_PIN_ACTION::PASS;
				CLOCK_SOURCE                       clock_source                         = CLOCK_SOURCE::IO_CLK_OVER_1;
				uint8_t                            output_compare_value_A               = 0xff;
				uint8_t                            output_compare_value_B               = 0xff;
				on_output_compare_match_func       on_output_compare_match_A            = nullptr;
				on_output_compare_match_func       on_output_compare_match_B            = nullptr;
				on_overflow_func                   on_overflow                          = nullptr;
			};

		public: // CONSTRUCTORS
			Timer0(void);
			Timer0(const Spec& spec);

		public: // GETTERS
			uint8_t count                    (void) const;
			uint8_t output_compare_register_A(void) const;
			uint8_t output_compare_register_B(void) const;
		public: // SETTERS
			void count                    (uint8_t new_count                    );
			void output_compare_register_A(uint8_t new_output_compare_register_A);
			void output_compare_register_B(uint8_t new_output_compare_register_B);
			//
			void on_compare_match_output_A_pin_action(ON_COMPARE_MATCH_OUTPUT_PIN_ACTION new_on_compare_match_output_A_pin_action);
			void on_compare_match_output_B_pin_action(ON_COMPARE_MATCH_OUTPUT_PIN_ACTION new_on_compare_match_output_B_pin_action);
			void mode(MODE new_mode);
			void clock_source(CLOCK_SOURCE new_clock_source);
			void on_output_compare_match_A(on_output_compare_match_func new_on_output_compare_match_A);
			void on_output_compare_match_B(on_output_compare_match_func new_on_output_compare_match_B);
			void on_overflow              (on_overflow_func             new_on_overflow              );

		public: // FUNCTIONS
			void init(const Spec& spec);

			void force_output_compare_A(void);
			void force_output_compare_B(void);

			void call_on_output_compare_match_A(void);
			void call_on_output_compare_match_B(void);
			void call_on_overflow              (void);

		private:
			on_output_compare_match_func m_on_output_compare_match_A;
			on_output_compare_match_func m_on_output_compare_match_B;
			on_overflow_func             m_on_overflow;
	};

	// GETTERS
	inline uint8_t Timer0::count                    (void) const { return TCNT0; }
	inline uint8_t Timer0::output_compare_register_A(void) const { return OCR0A; }
	inline uint8_t Timer0::output_compare_register_B(void) const { return OCR0B; }
	// SETTERS
	inline void Timer0::count                    (uint8_t new_count                    ) { TCNT0 = new_count;                     }
	inline void Timer0::output_compare_register_A(uint8_t new_output_compare_register_A) { OCR0A = new_output_compare_register_A; }
	inline void Timer0::output_compare_register_B(uint8_t new_output_compare_register_B) { OCR0B = new_output_compare_register_B; }
	//
	inline void Timer0::on_compare_match_output_A_pin_action(ON_COMPARE_MATCH_OUTPUT_PIN_ACTION new_on_compare_match_output_A_pin_action)
	{
		switch (new_on_compare_match_output_A_pin_action)
		{
			case ON_COMPARE_MATCH_OUTPUT_PIN_ACTION::PASS:
				CLEAR(TCCR0A, COM0A0);
				CLEAR(TCCR0A, COM0A1);
				break;
			case ON_COMPARE_MATCH_OUTPUT_PIN_ACTION::TOGGLE:
				SET  (TCCR0A, COM0A0);
				CLEAR(TCCR0A, COM0A1);
				break;
			case ON_COMPARE_MATCH_OUTPUT_PIN_ACTION::CLEAR:
				CLEAR(TCCR0A, COM0A0);
				SET  (TCCR0A, COM0A1);
				break;
			case ON_COMPARE_MATCH_OUTPUT_PIN_ACTION::SET:
				SET  (TCCR0A, COM0A0);
				SET  (TCCR0A, COM0A1);
				break;
		}
	}
	inline void Timer0::on_compare_match_output_B_pin_action(ON_COMPARE_MATCH_OUTPUT_PIN_ACTION new_on_compare_match_output_B_pin_action)
	{
		switch (new_on_compare_match_output_B_pin_action)
		{
			case ON_COMPARE_MATCH_OUTPUT_PIN_ACTION::PASS:
				CLEAR(TCCR0A, COM0B0);
				CLEAR(TCCR0A, COM0B1);
				break;
			case ON_COMPARE_MATCH_OUTPUT_PIN_ACTION::TOGGLE:
				SET  (TCCR0A, COM0B0);
				CLEAR(TCCR0A, COM0B1);
				break;
			case ON_COMPARE_MATCH_OUTPUT_PIN_ACTION::CLEAR:
				CLEAR(TCCR0A, COM0B0);
				SET  (TCCR0A, COM0B1);
				break;
			case ON_COMPARE_MATCH_OUTPUT_PIN_ACTION::SET:
				SET  (TCCR0A, COM0B0);
				SET  (TCCR0A, COM0B1);
				break;
		}
	}
	inline void Timer0::mode(MODE new_mode)
	{
		switch (new_mode)
		{
			case MODE::NORMAL:
				CLEAR(TCCR0A, WGM00);
				CLEAR(TCCR0A, WGM01);
				CLEAR(TCCR0A, WGM02);
				break;
			case MODE::CTC:
				CLEAR(TCCR0A, WGM00);
				SET  (TCCR0A, WGM01);
				CLEAR(TCCR0A, WGM02);
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
	}
	inline void Timer0::clock_source(CLOCK_SOURCE new_clock_source)
	{
		switch (new_clock_source)
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
	}
	inline void Timer0::on_output_compare_match_A(on_output_compare_match_func new_on_output_compare_match_A) { if (new_on_output_compare_match_A) SET(TIMSK0, OCIE0A); else CLEAR(TIMSK0, OCIE0A); m_on_output_compare_match_A = new_on_output_compare_match_A; }
	inline void Timer0::on_output_compare_match_B(on_output_compare_match_func new_on_output_compare_match_B) { if (new_on_output_compare_match_B) SET(TIMSK0, OCIE0A); else CLEAR(TIMSK0, OCIE0B); m_on_output_compare_match_B = new_on_output_compare_match_B; }
	inline void Timer0::on_overflow              (on_overflow_func             new_on_overflow              ) { if (new_on_overflow              ) SET(TIMSK0, TOIE0 ); else CLEAR(TIMSK0, TOIE0 ); m_on_overflow               = new_on_overflow;               }

	// FUNCTIONS
	inline void Timer0::force_output_compare_A(void) { SET(TCCR0B, FOC0A); }
	inline void Timer0::force_output_compare_B(void) { SET(TCCR0B, FOC0B); }
	//
	inline void Timer0::call_on_output_compare_match_A(void) { m_on_output_compare_match_A(); }
	inline void Timer0::call_on_output_compare_match_B(void) { m_on_output_compare_match_B(); }
	inline void Timer0::call_on_overflow              (void) { m_on_overflow              (); }

	extern Timer0 timer0;
}

#endif
