#ifndef _CORE_TIMER_BASE_H_
#define _CORE_TIMER_BASE_H_

#include <util.h>

#include "core.h"

namespace Core
{
	namespace TIMER
	{
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
			IO_CLK_OVER_32,
			IO_CLK_OVER_64,
			IO_CLK_OVER_128,
			IO_CLK_OVER_256,
			IO_CLK_OVER_1024,
			EXTERNAL_ON_FALLING_EDGE,
			EXTERNAL_ON_RISING_EDGE,
		};
		enum class TOP : uint8_t
		{
			MAX,
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
	}

	namespace Timer
	{
		struct Spec
		{
			TIMER::CLOCK_SOURCE                       clock_source;
			TIMER::MODE                               mode                                 = TIMER::MODE::NORMAL;
			TIMER::TOP                                top                                  = TIMER::TOP::MAX;
			TIMER::INPUT_CAPTURE                      input_capture                        = TIMER::INPUT_CAPTURE::DISABLED;
			TIMER::PIN_ACTION_ON_OUTPUT_COMPARE_MATCH pin_action_on_output_compare_match_A = TIMER::PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::PASS;
			TIMER::PIN_ACTION_ON_OUTPUT_COMPARE_MATCH pin_action_on_output_compare_match_B = TIMER::PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::PASS;
			interrupt_callback_func                   on_output_compare_match_A            = nullptr;
			interrupt_callback_func                   on_output_compare_match_B            = nullptr;
			interrupt_callback_func                   on_input_capture                     = nullptr;
			interrupt_callback_func                   on_overflow                          = nullptr;
		};

		template <typename count_t>
		class Base
		{
			protected: // CONSTRUCTORS
				Base(void) = default;
			protected: // DESTRUCTOR
				virtual ~Base(void) = default;

			public: // GETTERS
				virtual count_t count                 (void) const = 0;
				virtual count_t output_compare_value_A(void) const = 0;
				virtual count_t output_compare_value_B(void) const = 0;
				virtual count_t input_capture_value   (void) const = 0;
			public: // SETTERS
				virtual void count                 (count_t new_count                 ) = 0;
				virtual void output_compare_value_A(count_t new_output_compare_value_A) = 0;
				virtual void output_compare_value_B(count_t new_output_compare_value_B) = 0;
				virtual void input_capture_value   (count_t new_input_capture_value   ) = 0;
				//
				void mode(TIMER::MODE new_mode);
				void top(TIMER::TOP new_top);
				virtual void pin_action_on_output_compare_match_A(TIMER::PIN_ACTION_ON_OUTPUT_COMPARE_MATCH new_pin_action_on_output_compare_match_A) = 0;
				virtual void pin_action_on_output_compare_match_B(TIMER::PIN_ACTION_ON_OUTPUT_COMPARE_MATCH new_pin_action_on_output_compare_match_B) = 0;
				virtual void input_capture(TIMER::INPUT_CAPTURE new_input_capture) = 0;
				virtual void clock_source(TIMER::CLOCK_SOURCE new_clock_source) = 0;
				//
				virtual void on_output_compare_match_A(bool enable) = 0;
				virtual void on_output_compare_match_B(bool enable) = 0;
				virtual void on_overflow              (bool enable) = 0;
				virtual void on_input_capture         (bool enable) = 0;

			public: // METHODS
				void initialize(const Spec& spec);
				//
				void   pause(void);
				void unpause(void);
				//
				virtual void force_output_compare_A(void) = 0;
				virtual void force_output_compare_B(void) = 0;
				//
				template <TIMER::INTERRUPT interrupt> void  enable(void);
				template <TIMER::INTERRUPT interrupt> void disable(void);
				//
				template <TIMER::INTERRUPT interrupt> void call(void);
				//
				template <TIMER::INTERRUPT interrupt> void on(
					interrupt_callback_func interrupt_callback);

			protected:
				virtual void update_waveform_generation(void) = 0;

			protected:
				Spec m_spec;
		};

		// SETTERS
		template <typename count_t>
		inline void Base<count_t>::mode(TIMER::MODE new_mode)
		{
			m_spec.mode = new_mode;
			update_waveform_generation();
		}
		template <typename count_t>
		inline void Base<count_t>::top(TIMER::TOP new_top)
		{
			m_spec.top = new_top;
			update_waveform_generation();
		}

		// METHODS
		template <typename count_t>
		inline void Base<count_t>::initialize(const Spec& spec)
		{
			using namespace TIMER;

			mode(spec.mode);
			top (spec.top );

			input_capture(spec.input_capture);

			pin_action_on_output_compare_match_A(spec.pin_action_on_output_compare_match_A);
			pin_action_on_output_compare_match_B(spec.pin_action_on_output_compare_match_B);

			clock_source(spec.clock_source);

			on<INTERRUPT::ON_OUTPUT_COMPARE_MATCH_A>(spec.on_output_compare_match_A);
			on<INTERRUPT::ON_OUTPUT_COMPARE_MATCH_B>(spec.on_output_compare_match_B);
			on<INTERRUPT::ON_INPUT_CAPTURE         >(spec.on_input_capture         );
			on<INTERRUPT::ON_OVERFLOW              >(spec.on_overflow              );
		}
		//
		template <typename count_t>
		inline void Base<count_t>::pause(void)
		{
			using namespace TIMER;
			clock_source(CLOCK_SOURCE::NO);
		}
		template <typename count_t>
		inline void Base<count_t>::unpause(void)
		{
			using namespace TIMER;
			clock_source(m_spec.clock_source);
		}
		//
		template <typename count_t>
		template <TIMER::INTERRUPT interrupt>
		inline void Base<count_t>::enable(void)
		{
			using namespace TIMER;
			switch (interrupt)
			{
				case INTERRUPT::ON_OUTPUT_COMPARE_MATCH_A:
					on_output_compare_match_A(ENABLE);
					break;
				case INTERRUPT::ON_OUTPUT_COMPARE_MATCH_B:
					on_output_compare_match_B(ENABLE);
					break;
				case INTERRUPT::ON_INPUT_CAPTURE:
					on_input_capture(ENABLE);
					break;
				case INTERRUPT::ON_OVERFLOW:
					on_overflow(ENABLE);
					break;
			}
		}
		template <typename count_t>
		template <TIMER::INTERRUPT interrupt>
		inline void Base<count_t>::disable(void)
		{
			using namespace TIMER;
			switch (interrupt)
			{
				case INTERRUPT::ON_OUTPUT_COMPARE_MATCH_A:
					on_output_compare_match_A(DISABLE);
					break;
				case INTERRUPT::ON_OUTPUT_COMPARE_MATCH_B:
					on_output_compare_match_B(DISABLE);
					break;
				case INTERRUPT::ON_INPUT_CAPTURE:
					on_input_capture(DISABLE);
					break;
				case INTERRUPT::ON_OVERFLOW:
					on_overflow(DISABLE);
					break;
			}
		}
		//
		template <typename count_t>
		template <TIMER::INTERRUPT interrupt>
		inline void Base<count_t>::call(void)
		{
			using namespace TIMER;
			switch (interrupt)
			{
				case INTERRUPT::ON_OUTPUT_COMPARE_MATCH_A:
					m_spec.on_output_compare_match_A();
					break;
				case INTERRUPT::ON_OUTPUT_COMPARE_MATCH_B:
					m_spec.on_output_compare_match_B();
					break;
				case INTERRUPT::ON_INPUT_CAPTURE:
					m_spec.on_input_capture();
					break;
				case INTERRUPT::ON_OVERFLOW:
					m_spec.on_overflow();
					break;
			}
		}
		template <typename count_t>
		template <TIMER::INTERRUPT interrupt>
		inline void Base<count_t>::on(interrupt_callback_func interrupt_callback)
		{
			using namespace TIMER;
			switch (interrupt)
			{
				case INTERRUPT::ON_OUTPUT_COMPARE_MATCH_A:
					m_spec.on_output_compare_match_A = interrupt_callback;
					break;
				case INTERRUPT::ON_OUTPUT_COMPARE_MATCH_B:
					m_spec.on_output_compare_match_B = interrupt_callback;
					break;
				case INTERRUPT::ON_INPUT_CAPTURE:
					m_spec.on_input_capture = interrupt_callback;
					break;
				case INTERRUPT::ON_OVERFLOW:
					m_spec.on_overflow = interrupt_callback;
					break;
			}
		}
	}
}

#endif
