#ifndef _TIMER_TIMER1_H_
#define _TIMER_TIMER1_H_

#include <avr/io.h>

#include <util.h>

#include "timer/itimer.h"

class Timer1: virtual public ITimer
{
	public: // TYPES
		enum class MODE                    : uint8_t { NORMAL, CTC, FAST_PWM, PHASE_CORRECT_PWM, PHASE_AND_FREQUENCY_CORRECT_PWM };
		enum class INPUT_CAPTURE           : bool    { ENABLED, DISABLED };
		enum class ON_COMPARE_MATCH_OUTPUT : uint8_t { PASS, TOGGLE, CLEAR, SET };
		enum class CLOCK_SOURCE            : uint8_t { IO_CLK_OVER_1,  IO_CLK_OVER_8,  IO_CLK_OVER_64, IO_CLK_OVER_256, IO_CLK_OVER_1024, EXTERNAL_ON_FALLING_EDGE, EXTERNAL_ON_RISING_EDGE };
		using on_output_compare_match_func = void (*)(void);
		using on_overflow_func             = void (*)(void);

		struct Init
		{
			MODE                         mode                      = MODE::NORMAL;
			INPUT_CAPTURE                input_capture             = INPUT_CAPTURE::DISABLED;
			ON_COMPARE_MATCH_OUTPUT      on_compare_match_output_A = ON_COMPARE_MATCH_OUTPUT::PASS;
			ON_COMPARE_MATCH_OUTPUT      on_compare_match_output_B = ON_COMPARE_MATCH_OUTPUT::PASS;
			CLOCK_SOURCE                 clock_source              = CLOCK_SOURCE::IO_CLK_OVER_1;
			uint16_t                     output_compare_value_A    = 0xffff;
			uint16_t                     output_compare_value_B    = 0xffff;
			on_output_compare_match_func on_output_compare_match_A = nullptr;
			on_output_compare_match_func on_output_compare_match_B = nullptr;
			on_overflow_func             on_overflow               = nullptr;
		};

	public: // CONSTRUCTORS
		Timer1(void);
		Timer1(const Init& init_struct);

	public: // PUBLIC VARIABLES
		on_output_compare_match_func on_output_compare_match_A;
		on_output_compare_match_func on_output_compare_match_B;
		on_overflow_func             on_overflow;

	public: // GETTERS
		uint16_t count                    (void) const;
		uint16_t output_compare_register_A(void) const;
		uint16_t output_compare_register_B(void) const;
	public: // SETTERS
		void count                    (uint16_t new_count                    );
		void output_compare_register_A(uint16_t new_output_compare_register_A);
		void output_compare_register_B(uint16_t new_output_compare_register_B);

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
inline uint16_t Timer1::count                    (void) const { return TCNT1; }
inline uint16_t Timer1::output_compare_register_A(void) const { return OCR1A; }
inline uint16_t Timer1::output_compare_register_B(void) const { return OCR1B; }
// SETTERS
inline void Timer1::count                    (uint16_t new_count                    ) { TCNT1 = new_count;                     }
inline void Timer1::output_compare_register_A(uint16_t new_output_compare_register_A) { OCR1A = new_output_compare_register_A; }
inline void Timer1::output_compare_register_B(uint16_t new_output_compare_register_B) { OCR1B = new_output_compare_register_B; }

// FUNCTIONS
inline void Timer1::force_output_compare_A(void) { SET(TCCR1B, FOC1A); }
inline void Timer1::force_output_compare_B(void) { SET(TCCR1B, FOC1B); }

inline void Timer1:: enable_output_compare_match_A_interrupt(void) { SET  (TIMSK1, OCIE1A); }
inline void Timer1::disable_output_compare_match_A_interrupt(void) { CLEAR(TIMSK1, OCIE1A); }
inline void Timer1:: enable_output_compare_match_B_interrupt(void) { SET  (TIMSK1, OCIE1B); }
inline void Timer1::disable_output_compare_match_B_interrupt(void) { CLEAR(TIMSK1, OCIE1B); }
inline void Timer1::               enable_overflow_interrupt(void) { SET  (TIMSK1, TOIE1 ); }
inline void Timer1::              disable_overflow_interrupt(void) { CLEAR(TIMSK1, TOIE1 ); }

extern Timer1 timer1;

#endif
