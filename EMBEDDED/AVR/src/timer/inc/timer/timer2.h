#ifndef _TIMER_TIMER2_H_
#define _TIMER_TIMER2_H_

#include "timer/base.h"

namespace Timer
{
	class Timer2: virtual public Timer::Base
	{
		public: // TYPES
			enum class MODE : uint8_t
			{
				NORMAL,
				CTC,
				FAST_PWM,
				PHASE_CORRECT_PWM
			};
			enum class ON_COMPARE_MATCH_OUTPUT : uint8_t
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
				MODE                         mode                      = MODE::NORMAL;
				ON_COMPARE_MATCH_OUTPUT      on_compare_match_output_A = ON_COMPARE_MATCH_OUTPUT::PASS;
				ON_COMPARE_MATCH_OUTPUT      on_compare_match_output_B = ON_COMPARE_MATCH_OUTPUT::PASS;
				CLOCK_SOURCE                 clock_source              = CLOCK_SOURCE::IO_CLK_OVER_1;
				uint8_t                      output_compare_value_A    = 0xff;
				uint8_t                      output_compare_value_B    = 0xff;
				on_output_compare_match_func on_output_compare_match_A = nullptr;
				on_output_compare_match_func on_output_compare_match_B = nullptr;
				on_overflow_func             on_overflow               = nullptr;
			};

		public: // CONSTRUCTORS
			Timer2(void) = default;
			Timer2(const Spec& spec);
		public: // DESTRUCTOR
			~Timer2(void) = default;

		public: // PUBLIC VARIABLES
			on_output_compare_match_func on_output_compare_match_A;
			on_output_compare_match_func on_output_compare_match_B;
			on_overflow_func             on_overflow;

		public: // GETTERS
			uint8_t count                    (void) const;
			uint8_t output_compare_register_A(void) const;
			uint8_t output_compare_register_B(void) const;
		public: // SETTERS
			void count                    (uint8_t new_count                    );
			void output_compare_register_A(uint8_t new_output_compare_register_A);
			void output_compare_register_B(uint8_t new_output_compare_register_B);
			//
			void clock_source(CLOCK_SOURCE new_clock_source);

		public: // METHODS
			void init(const Spec& spec);
			//
			void   pause(void) override;
			void unpause(void) override;
			//
			void force_output_compare_A(void);
			void force_output_compare_B(void);
			//
			void  enable_output_compare_match_A_interrupt(void);
			void disable_output_compare_match_A_interrupt(void);
			void  enable_output_compare_match_B_interrupt(void);
			void disable_output_compare_match_B_interrupt(void);
			void                enable_overflow_interrupt(void);
			void               disable_overflow_interrupt(void);

		private:
			Spec m_spec;
	};

	// GETTERS
	inline uint8_t Timer2::count                    (void) const { return TCNT2; }
	inline uint8_t Timer2::output_compare_register_A(void) const { return OCR2A; }
	inline uint8_t Timer2::output_compare_register_B(void) const { return OCR2B; }
	// SETTERS
	inline void Timer2::count                    (uint8_t new_count                    ) { TCNT2 = new_count;                     }
	inline void Timer2::output_compare_register_A(uint8_t new_output_compare_register_A) { OCR2A = new_output_compare_register_A; }
	inline void Timer2::output_compare_register_B(uint8_t new_output_compare_register_B) { OCR2B = new_output_compare_register_B; }

	// METHODS
	inline void Timer2::  pause(void) { clock_source(CLOCK_SOURCE::NO); }
	inline void Timer2::unpause(void) { clock_source(m_spec.clock_source); }
	//
	inline void Timer2::force_output_compare_A(void) { SET(TCCR2B, FOC2A); }
	inline void Timer2::force_output_compare_B(void) { SET(TCCR2B, FOC2B); }
	//
	inline void Timer2:: enable_output_compare_match_A_interrupt(void) { SET  (TIMSK2, OCIE2A); }
	inline void Timer2::disable_output_compare_match_A_interrupt(void) { CLEAR(TIMSK2, OCIE2A); }
	inline void Timer2:: enable_output_compare_match_B_interrupt(void) { SET  (TIMSK2, OCIE2B); }
	inline void Timer2::disable_output_compare_match_B_interrupt(void) { CLEAR(TIMSK2, OCIE2B); }
	inline void Timer2::               enable_overflow_interrupt(void) { SET  (TIMSK2, TOIE2 ); }
	inline void Timer2::              disable_overflow_interrupt(void) { CLEAR(TIMSK2, TOIE2 ); }

	extern Timer2 timer2;
}

#endif
