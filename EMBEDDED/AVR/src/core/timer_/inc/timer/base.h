#ifndef _TIMER_BASE_H_
#define _TIMER_BASE_H_

#include <util.h>

namespace Timer
{
	class Base
	{
		public: // TYPES
			enum class MODE : uint8_t
			{
				NORMAL,
				CTC,
				FAST_PWM,
				PHASE_CORRECT_PWM,
				PHASE_AND_FREQUENCY_CORRECT_PWM,
			};
			enum class INPUT_CAPTURE : uint8_t
			{
				DISABLED,
				ON_FALLING_EDGE,
				ON_RISING_EDGE,
			};
			enum class PIN_ACTION_ON_OUTPUT_COMPARE_MATCH : uint8_t
			{
				PASS,
				TOGGLE,
				CLEAR,
				SET,
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
				EXTERNAL_ON_RISING_EDGE,
			};
			enum class TOP : uint8_t
			{
				XFFFF,
				X00FF,
				X01FF,
				X03FF,
				INPUT_CAPTURE_VALUE,
				OUTPUT_COMPARE_VALUE_A,
			};
			enum class INTERRUPT : uint8_t
			{
				ON_OUTPUT_COMPARE_MATCH_A,
				ON_OUTPUT_COMPARE_MATCH_B,
				ON_INPUT_CAPTURE,
				ON_OVERFLOW,
			};
			using interrupt_callback_func = void (*)(void);
			struct Spec
			{
				MODE                               mode                                 = MODE::NORMAL;
				TOP                                top                                  = TOP::MAX;
				INPUT_CAPTURE                      input_capture                        = INPUT_CAPTURE::DISABLED;
				PIN_ACTION_ON_OUTPUT_COMPARE_MATCH pin_action_on_output_compare_match_A = PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::PASS;
				PIN_ACTION_ON_OUTPUT_COMPARE_MATCH pin_action_on_output_compare_match_B = PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::PASS;
				CLOCK_SOURCE                       clock_source;
				interrupt_callback_func            on_output_compare_match_A            = nullptr;
				interrupt_callback_func            on_output_compare_match_B            = nullptr;
				interrupt_callback_func            on_input_capture                     = nullptr;
				interrupt_callback_func            on_overflow                          = nullptr;
			};

		public: // CONSTRUCTORS
			Base(void) = default;
		public: // DESTRUCTOR
			virtual ~Base(void) = default;

		public: // METHODS
			virtual void   pause(void) = 0;
			virtual void unpause(void) = 0;
	};
}

#endif
