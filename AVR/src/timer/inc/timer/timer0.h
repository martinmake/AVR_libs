#ifndef _TIMER_TIMER0_H_
#define _TIMER_TIMER0_H_

#include <avr/io.h>

#include <util.h>

#include "timer/itimer.h"

class Timer0: virtual public ITimer
{
	public: // TYPES
		enum class MODE                    : uint8_t { NON_PWM, FAST_PWM, PHASE_CORRECT_PWM };
		enum class ON_COMPARE_MATCH_OUTPUT : uint8_t { PASS, TOGGLE, CLEAR, SET };
		enum class CLOCK_SOURCE            : uint8_t { IO_CLK_OVER_1,  IO_CLK_OVER_8,  IO_CLK_OVER_64, IO_CLK_OVER_256, IO_CLK_OVER_1024, EXTERNAL_ON_FALLING_EDGE, EXTERNAL_ON_RISING_EDGE };
		using on_output_compare_match_func = void (*)(void);
		using on_overflow_func             = void (*)(void);

		struct Init
		{
			MODE                         mode                      = MODE::NON_PWM;
			ON_COMPARE_MATCH_OUTPUT      on_compare_match_output_A = ON_COMPARE_MATCH_OUTPUT::PASS;
			ON_COMPARE_MATCH_OUTPUT      on_compare_match_output_B = ON_COMPARE_MATCH_OUTPUT::PASS;
			CLOCK_SOURCE                 clock_source              = CLOCK_SOURCE::IO_CLK_OVER_1;
			bool                         ctc                       = false;
			uint8_t                      output_compare_value_A    = 0xff;
			uint8_t                      output_compare_value_B    = 0xff;
			on_output_compare_match_func on_output_compare_match_A = nullptr;
			on_output_compare_match_func on_output_compare_match_B = nullptr;
			on_overflow_func             on_overflow               = nullptr;
		};

	public: // CONSTRUCTORS
		Timer0(void);
		Timer0(const Init& init_struct);

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

	public: // FUNCTIONS
		void init(const Init& init_struct);

		void force_output_compare_A(void);
		void force_output_compare_B(void);

		void  enable_output_compare_match_A_interrupt(void);
		void disable_output_compare_match_A_interrupt(void);
		void  enable_output_compare_match_B_interrupt(void);
		void disable_output_compare_match_B_interrupt(void);
		void                enable_overflow_interrupt(void);
		void               disable_overflow_interrupt(void);
};

// GETTERS
inline uint8_t Timer0::count                    (void) const { return TCNT0; }
inline uint8_t Timer0::output_compare_register_A(void) const { return OCR0A; }
inline uint8_t Timer0::output_compare_register_B(void) const { return OCR0B; }
// SETTERS
inline void Timer0::count                    (uint8_t new_count                    ) { TCNT0 = new_count;                     }
inline void Timer0::output_compare_register_A(uint8_t new_output_compare_register_A) { OCR0A = new_output_compare_register_A; }
inline void Timer0::output_compare_register_B(uint8_t new_output_compare_register_B) { OCR0B = new_output_compare_register_B; }

// FUNCTIONS
inline void Timer0::force_output_compare_A(void) { SET(TCCR0B, FOC0A); }
inline void Timer0::force_output_compare_B(void) { SET(TCCR0B, FOC0B); }

inline void Timer0:: enable_output_compare_match_A_interrupt(void) { SET  (TIMSK0, OCIE0A); }
inline void Timer0::disable_output_compare_match_A_interrupt(void) { CLEAR(TIMSK0, OCIE0A); }
inline void Timer0:: enable_output_compare_match_B_interrupt(void) { SET  (TIMSK0, OCIE0B); }
inline void Timer0::disable_output_compare_match_B_interrupt(void) { CLEAR(TIMSK0, OCIE0B); }
inline void Timer0::               enable_overflow_interrupt(void) { SET  (TIMSK0, TOIE0 ); }
inline void Timer0::              disable_overflow_interrupt(void) { CLEAR(TIMSK0, TOIE0 ); }

extern Timer0 timer0;

#endif
