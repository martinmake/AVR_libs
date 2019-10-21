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
			enum class PIN_ACTION_ON_OUTPUT_COMPARE_MATCH : uint8_t
			{
				PASS,
				TOGGLE,
				CLEAR,
				SET
			};
			enum class CLOCK_SOURCE : uint8_t
			{
				NO,
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
				PIN_ACTION_ON_OUTPUT_COMPARE_MATCH pin_action_on_output_compare_match_A = PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::PASS;
				PIN_ACTION_ON_OUTPUT_COMPARE_MATCH pin_action_on_output_compare_match_B = PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::PASS;
				CLOCK_SOURCE                       clock_source                         = CLOCK_SOURCE::IO_CLK_OVER_1;
				uint8_t                            output_compare_value_A               = 0xff;
				uint8_t                            output_compare_value_B               = 0xff;
				on_output_compare_match_func       on_output_compare_match_A            = nullptr;
				on_output_compare_match_func       on_output_compare_match_B            = nullptr;
				on_overflow_func                   on_overflow                          = nullptr;
			};

		public: // CONSTRUCTORS
			Timer0(void) = default;
			Timer0(const Spec& spec);
		public: // DESTRUCTOR
			~Timer0(void) = default;

		public: // GETTERS
			uint8_t count                    (void) const;
			uint8_t output_compare_register_A(void) const;
			uint8_t output_compare_register_B(void) const;
		public: // SETTERS
			void count                    (uint8_t new_count                    );
			void output_compare_register_A(uint8_t new_output_compare_register_A);
			void output_compare_register_B(uint8_t new_output_compare_register_B);
			//
			void pin_action_on_output_compare_match_A(PIN_ACTION_ON_OUTPUT_COMPARE_MATCH new_pin_action_on_output_compare_match_A);
			void pin_action_on_output_compare_match_B(PIN_ACTION_ON_OUTPUT_COMPARE_MATCH new_pin_action_on_output_compare_match_B);
			void mode(MODE new_mode);
			void clock_source(CLOCK_SOURCE new_clock_source);
			void on_output_compare_match_A(on_output_compare_match_func new_on_output_compare_match_A);
			void on_output_compare_match_B(on_output_compare_match_func new_on_output_compare_match_B);
			void on_overflow              (on_overflow_func             new_on_overflow              );
			//
			void  enable_output_compare_match_A_interrupt(void);
			void disable_output_compare_match_A_interrupt(void);
			void  enable_output_compare_match_B_interrupt(void);
			void disable_output_compare_match_B_interrupt(void);
			void                enable_overflow_interrupt(void);
			void               disable_overflow_interrupt(void);

		public: // METHODS
			void init(const Spec& spec);
			//
			void pause  (void) override;
			void unpause(void) override;
			//
			void force_output_compare_A(void);
			void force_output_compare_B(void);
			//
			void call_on_output_compare_match_A(void);
			void call_on_output_compare_match_B(void);
			void call_on_overflow              (void);

		private:
			CLOCK_SOURCE m_clock_source;
			//
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
	inline void Timer0::pin_action_on_output_compare_match_A(
			PIN_ACTION_ON_OUTPUT_COMPARE_MATCH new_pin_action_on_output_compare_match_A)
	{
		switch (new_pin_action_on_output_compare_match_A)
		{
			case PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::PASS:
				CLEAR(TCCR0A, COM0A0);
				CLEAR(TCCR0A, COM0A1);
				break;
			case PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::TOGGLE:
				SET  (TCCR0A, COM0A0);
				CLEAR(TCCR0A, COM0A1);
				break;
			case PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::CLEAR:
				CLEAR(TCCR0A, COM0A0);
				SET  (TCCR0A, COM0A1);
				break;
			case PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::SET:
				SET  (TCCR0A, COM0A0);
				SET  (TCCR0A, COM0A1);
				break;
		}
	}
	inline void Timer0::pin_action_on_output_compare_match_B(
		PIN_ACTION_ON_OUTPUT_COMPARE_MATCH new_pin_action_on_output_compare_match_B)
	{
		switch (new_pin_action_on_output_compare_match_B)
		{
			case PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::PASS:
				CLEAR(TCCR0A, COM0B0);
				CLEAR(TCCR0A, COM0B1);
				break;
			case PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::TOGGLE:
				SET  (TCCR0A, COM0B0);
				CLEAR(TCCR0A, COM0B1);
				break;
			case PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::CLEAR:
				CLEAR(TCCR0A, COM0B0);
				SET  (TCCR0A, COM0B1);
				break;
			case PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::SET:
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
		m_clock_source = new_clock_source;

		switch (new_clock_source)
		{
			case CLOCK_SOURCE::NO:
				CLEAR(TCCR0B, CS00);
				CLEAR(TCCR0B, CS01);
				CLEAR(TCCR0B, CS02);
				break;
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
	inline void Timer0::on_output_compare_match_A(
		on_output_compare_match_func new_on_output_compare_match_A)
	{
		m_on_output_compare_match_A = new_on_output_compare_match_A;
		if (m_on_output_compare_match_A)
			enable_output_compare_match_A_interrupt();
		else
			disable_output_compare_match_A_interrupt();
	}
	inline void Timer0::on_output_compare_match_B(
		on_output_compare_match_func new_on_output_compare_match_B)
	{
		m_on_output_compare_match_B = new_on_output_compare_match_B;
		if (m_on_output_compare_match_B)
			enable_output_compare_match_B_interrupt();
		else
			disable_output_compare_match_B_interrupt();
	}
	inline void Timer0::on_overflow(on_overflow_func new_on_overflow )
	{
		m_on_overflow = new_on_overflow;
		if (m_on_overflow)  enable_overflow_interrupt();
		else               disable_overflow_interrupt();
	}
	//
	inline void Timer0:: enable_output_compare_match_A_interrupt(void) { SET  (TIMSK0, OCIE0A); }
	inline void Timer0::disable_output_compare_match_A_interrupt(void) { CLEAR(TIMSK0, OCIE0A); }
	inline void Timer0:: enable_output_compare_match_B_interrupt(void) { SET  (TIMSK0, OCIE0B); }
	inline void Timer0::disable_output_compare_match_B_interrupt(void) { CLEAR(TIMSK0, OCIE0B); }
	inline void Timer0::               enable_overflow_interrupt(void) { SET  (TIMSK0, TOIE0 ); }
	inline void Timer0::              disable_overflow_interrupt(void) { CLEAR(TIMSK0, TOIE0 ); }

	// METHODS
	inline void Timer0::  pause(void) { clock_source(CLOCK_SOURCE::NO); }
	inline void Timer0::unpause(void) { clock_source(m_clock_source  ); }
	//
	inline void Timer0::force_output_compare_A(void) { SET(TCCR0B, FOC0A); }
	inline void Timer0::force_output_compare_B(void) { SET(TCCR0B, FOC0B); }
	//
	inline void Timer0::call_on_output_compare_match_A(void)
	{ m_on_output_compare_match_A(); }
	inline void Timer0::call_on_output_compare_match_B(void)
	{ m_on_output_compare_match_B(); }
	inline void Timer0::call_on_overflow(void) { m_on_overflow(); }

	extern Timer0 timer0;
}

#endif
