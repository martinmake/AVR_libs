#ifndef _TIMER_TIMER1_H_
#define _TIMER_TIMER1_H_

#include <avr/io.h>

#include <util.h>

#include "timer/base.h"

namespace Timer
{
	class Timer1: virtual public Timer::Base
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
				TOP                                top                                  = TOP::XFFFF;
				INPUT_CAPTURE                      input_capture                        = INPUT_CAPTURE::DISABLED;
				PIN_ACTION_ON_OUTPUT_COMPARE_MATCH pin_action_on_output_compare_match_A = PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::PASS;
				PIN_ACTION_ON_OUTPUT_COMPARE_MATCH pin_action_on_output_compare_match_B = PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::PASS;
				CLOCK_SOURCE                       clock_source                         = CLOCK_SOURCE::IO_CLK_OVER_1;
				interrupt_callback_func            on_output_compare_match_A            = nullptr;
				interrupt_callback_func            on_output_compare_match_B            = nullptr;
				interrupt_callback_func            on_input_capture                     = nullptr;
				interrupt_callback_func            on_overflow                          = nullptr;
			};

		public: // CONSTRUCTORS
			Timer1(void) = default;
			Timer1(const Spec& spec);
		public: // DESTRUCTOR
			~Timer1(void) = default;

		public: // GETTERS
			uint16_t count                 (void) const;
			uint16_t output_compare_value_A(void) const;
			uint16_t output_compare_value_B(void) const;
			uint16_t input_capture_value   (void) const;
		public: // SETTERS
			void count                 (uint16_t new_count                 );
			void output_compare_value_A(uint16_t new_output_compare_value_A);
			void output_compare_value_B(uint16_t new_output_compare_value_B);
			void input_capture_value   (uint16_t new_input_capture_value   );
			//
			void pin_action_on_output_compare_match_A(PIN_ACTION_ON_OUTPUT_COMPARE_MATCH new_pin_action_on_output_compare_match_A);
			void pin_action_on_output_compare_match_B(PIN_ACTION_ON_OUTPUT_COMPARE_MATCH new_pin_action_on_output_compare_match_B);
			void mode(MODE new_mode);
			void top(TOP new_top);
			void input_capture(INPUT_CAPTURE new_input_capture);
			void clock_source(CLOCK_SOURCE new_clock_source);

		public: // METHODS
			void init(const Spec& spec);
			//
			void pause  (void) override;
			void unpause(void) override;
			//
			void force_output_compare_A(void);
			void force_output_compare_B(void);
			//
			template <INTERRUPT interrupt> void call(void);
			//
			template <INTERRUPT interrupt> void  enable(void);
			template <INTERRUPT interrupt> void disable(void);
			//
			template <INTERRUPT interrupt> void on(interrupt_callback_func interrupt_callback);

		private:
			Spec m_spec;

		private:
			void update_waveform_generation(void);
	};

	// GETTERS
	inline uint16_t Timer1::count                 (void) const { return TCNT1; }
	inline uint16_t Timer1::output_compare_value_A(void) const { return OCR1A; }
	inline uint16_t Timer1::output_compare_value_B(void) const { return OCR1B; }
	inline uint16_t Timer1::input_capture_value   (void) const { return ICR1;  }
	// SETTERS
	inline void Timer1::count                 (uint16_t new_count                 ) { TCNT1 = new_count;                  }
	inline void Timer1::output_compare_value_A(uint16_t new_output_compare_value_A) { OCR1A = new_output_compare_value_A; }
	inline void Timer1::output_compare_value_B(uint16_t new_output_compare_value_B) { OCR1B = new_output_compare_value_B; }
	inline void Timer1::input_capture_value   (uint16_t new_input_capture_value   ) { ICR1  = new_input_capture_value;    }
	//
	inline void Timer1::pin_action_on_output_compare_match_A(
		PIN_ACTION_ON_OUTPUT_COMPARE_MATCH new_pin_action_on_output_compare_match_A)
	{
		m_spec.pin_action_on_output_compare_match_A = new_pin_action_on_output_compare_match_A;

		switch (new_pin_action_on_output_compare_match_A)
		{
			case PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::PASS:
				CLEAR(TCCR1A, COM1A0);
				CLEAR(TCCR1A, COM1A1);
				break;
			case PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::TOGGLE:
				SET  (TCCR1A, COM1A0);
				CLEAR(TCCR1A, COM1A1);
				break;
			case PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::CLEAR:
				CLEAR(TCCR1A, COM1A0);
				SET  (TCCR1A, COM1A1);
				break;
			case PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::SET:
				SET  (TCCR1A, COM1A0);
				SET  (TCCR1A, COM1A1);
				break;
		}
	}
	inline void Timer1::pin_action_on_output_compare_match_B(
		PIN_ACTION_ON_OUTPUT_COMPARE_MATCH new_pin_action_on_output_compare_match_B)
	{
		m_spec.pin_action_on_output_compare_match_B = new_pin_action_on_output_compare_match_B;

		switch (new_pin_action_on_output_compare_match_B)
		{
			case PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::PASS:
				CLEAR(TCCR1A, COM1B0);
				CLEAR(TCCR1A, COM1B1);
				break;
			case PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::TOGGLE:
				SET  (TCCR1A, COM1B0);
				CLEAR(TCCR1A, COM1B1);
				break;
			case PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::CLEAR:
				CLEAR(TCCR1A, COM1B0);
				SET  (TCCR1A, COM1B1);
				break;
			case PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::SET:
				SET  (TCCR1A, COM1B0);
				SET  (TCCR1A, COM1B1);
				break;
		}
	}
	inline void Timer1::input_capture(INPUT_CAPTURE new_input_capture)
	{
		m_spec.input_capture = new_input_capture;

		switch (new_input_capture)
		{
			case INPUT_CAPTURE::DISABLED:
				CLEAR(TIMSK1, ICIE1);
				break;
			case INPUT_CAPTURE::ON_FALLING_EDGE:
				SET  (TIMSK1, ICIE1);
				CLEAR(TCCR1B, ICES1);
				break;
			case INPUT_CAPTURE::ON_RISING_EDGE:
				SET  (TIMSK1, ICIE1);
				SET  (TCCR1B, ICES1);
				break;
		}
	}
	inline void Timer1::mode(MODE new_mode)
	{
		m_spec.mode = new_mode;

		update_waveform_generation();
	}
	inline void Timer1::top(TOP new_top)
	{
		m_spec.top = new_top;

		update_waveform_generation();
	}
	inline void Timer1::clock_source(CLOCK_SOURCE new_clock_source)
	{
		m_spec.clock_source = new_clock_source;

		switch (new_clock_source)
		{
			case CLOCK_SOURCE::NO:
				CLEAR(TCCR2B, CS10);
				CLEAR(TCCR2B, CS11);
				CLEAR(TCCR2B, CS12);
				break;
			case CLOCK_SOURCE::IO_CLK_OVER_1:
				SET  (TCCR1B, CS10);
				CLEAR(TCCR1B, CS11);
				CLEAR(TCCR1B, CS12);
				break;
			case CLOCK_SOURCE::IO_CLK_OVER_8:
				SET  (TCCR1B, CS10);
				CLEAR(TCCR1B, CS11);
				CLEAR(TCCR1B, CS12);
				break;
			case CLOCK_SOURCE::IO_CLK_OVER_64:
				SET  (TCCR1B, CS10);
				SET  (TCCR1B, CS11);
				CLEAR(TCCR1B, CS12);
				break;
			case CLOCK_SOURCE::IO_CLK_OVER_256:
				CLEAR(TCCR1B, CS10);
				CLEAR(TCCR1B, CS11);
				SET  (TCCR1B, CS12);
				break;
			case CLOCK_SOURCE::IO_CLK_OVER_1024:
				SET  (TCCR1B, CS10);
				CLEAR(TCCR1B, CS11);
				SET  (TCCR1B, CS12);
				break;
			case CLOCK_SOURCE::EXTERNAL_ON_FALLING_EDGE:
				CLEAR(TCCR1B, CS10);
				SET  (TCCR1B, CS11);
				SET  (TCCR1B, CS12);
				break;
			case CLOCK_SOURCE::EXTERNAL_ON_RISING_EDGE:
				SET  (TCCR1B, CS10);
				SET  (TCCR1B, CS11);
				SET  (TCCR1B, CS12);
				break;
		}
	}
	template <Timer1::INTERRUPT interrupt>
	inline void Timer1::on(interrupt_callback_func interrupt_callback)
	{
		switch (interrupt)
		{
			case INTERRUPT::ON_OUTPUT_COMPARE_MATCH_A:
				m_spec.on_output_compare_match_A = interrupt_callback;
				if (interrupt_callback)
					enable<INTERRUPT::ON_OUTPUT_COMPARE_MATCH_A>();
				break;
			case INTERRUPT::ON_OUTPUT_COMPARE_MATCH_B:
				m_spec.on_output_compare_match_B = interrupt_callback;
				if (interrupt_callback)
					enable<INTERRUPT::ON_OUTPUT_COMPARE_MATCH_B>();
				break;
			case INTERRUPT::ON_INPUT_CAPTURE:
				m_spec.on_input_capture = interrupt_callback;
				if (interrupt_callback)
					enable<INTERRUPT::ON_INPUT_CAPTURE>();
				break;
			case INTERRUPT::ON_OVERFLOW:
				m_spec.on_overflow = interrupt_callback;
				if (interrupt_callback)
					enable<INTERRUPT::ON_OVERFLOW>();
				break;
		}
	}

	// METHODS
	inline void Timer1::update_waveform_generation(void)
	{
		switch (m_spec.mode)
		{
			case MODE::NORMAL: switch (m_spec.top)
			{
				case TOP::XFFFF:
					CLEAR(TCCR1A, WGM10);
					CLEAR(TCCR1A, WGM11);
					CLEAR(TCCR1B, WGM12);
					CLEAR(TCCR1B, WGM13);
					break;
				default: break;
			} break;
			case MODE::CTC: switch (m_spec.top)
			{
				case TOP::OUTPUT_COMPARE_VALUE_A:
					CLEAR(TCCR1A, WGM10);
					CLEAR(TCCR1A, WGM11);
					SET  (TCCR1B, WGM12);
					CLEAR(TCCR1B, WGM13);
					break;
				case TOP::INPUT_CAPTURE_VALUE:
					CLEAR(TCCR1A, WGM10);
					CLEAR(TCCR1A, WGM11);
					SET  (TCCR1B, WGM12);
					SET  (TCCR1B, WGM13);
					break;
				default: break;
			} break;
			case MODE::FAST_PWM: switch (m_spec.top)
			{
				case TOP::OUTPUT_COMPARE_VALUE_A:
					SET  (TCCR1A, WGM10);
					SET  (TCCR1A, WGM11);
					SET  (TCCR1B, WGM12);
					SET  (TCCR1B, WGM13);
					break;
				case TOP::INPUT_CAPTURE_VALUE:
					CLEAR(TCCR1A, WGM10);
					SET  (TCCR1A, WGM11);
					SET  (TCCR1B, WGM12);
					SET  (TCCR1B, WGM13);
					break;
				case TOP::X00FF:
					SET  (TCCR1A, WGM10);
					CLEAR(TCCR1A, WGM11);
					SET  (TCCR1B, WGM12);
					CLEAR(TCCR1B, WGM13);
					break;
				case TOP::X01FF:
					CLEAR(TCCR1A, WGM10);
					SET  (TCCR1A, WGM11);
					SET  (TCCR1B, WGM12);
					CLEAR(TCCR1B, WGM13);
					break;
				case TOP::X03FF:
					SET  (TCCR1A, WGM10);
					SET  (TCCR1A, WGM11);
					SET  (TCCR1B, WGM12);
					CLEAR(TCCR1B, WGM13);
					break;
				default: break;
			} break;
			case MODE::PHASE_CORRECT_PWM: switch (m_spec.top)
			{
				case TOP::OUTPUT_COMPARE_VALUE_A:
					SET  (TCCR1A, WGM10);
					SET  (TCCR1A, WGM11);
					CLEAR(TCCR1B, WGM12);
					SET  (TCCR1B, WGM13);
					break;
				case TOP::INPUT_CAPTURE_VALUE:
					CLEAR(TCCR1A, WGM10);
					SET  (TCCR1A, WGM11);
					CLEAR(TCCR1B, WGM12);
					SET  (TCCR1B, WGM13);
					break;
				case TOP::X00FF:
					CLEAR(TCCR1A, WGM10);
					CLEAR(TCCR1A, WGM11);
					CLEAR(TCCR1B, WGM12);
					SET  (TCCR1B, WGM13);
					break;
				case TOP::X01FF:
					CLEAR(TCCR1A, WGM10);
					SET  (TCCR1A, WGM11);
					CLEAR(TCCR1B, WGM12);
					CLEAR(TCCR1B, WGM13);
					break;
				case TOP::X03FF:
					SET  (TCCR1A, WGM10);
					SET  (TCCR1A, WGM11);
					CLEAR(TCCR1B, WGM12);
					CLEAR(TCCR1B, WGM13);
					break;
				default: break;
			} break;
			case MODE::PHASE_AND_FREQUENCY_CORRECT_PWM: switch (m_spec.top)
			{
				case TOP::OUTPUT_COMPARE_VALUE_A:
					SET  (TCCR1A, WGM10);
					CLEAR(TCCR1A, WGM11);
					CLEAR(TCCR1B, WGM12);
					SET  (TCCR1B, WGM13);
					break;
				case TOP::INPUT_CAPTURE_VALUE:
					CLEAR(TCCR1A, WGM10);
					CLEAR(TCCR1A, WGM11);
					CLEAR(TCCR1B, WGM12);
					SET  (TCCR1B, WGM13);
					break;
				default: break;
			} break;
		}
	}
	template <Timer1::INTERRUPT interrupt>
	inline void Timer1::enable(void)
	{
		switch (interrupt)
		{
			case INTERRUPT::ON_OUTPUT_COMPARE_MATCH_A:
				 SET(TIMSK1, OCIE1A);
			case INTERRUPT::ON_OUTPUT_COMPARE_MATCH_B:
				 SET(TIMSK1, OCIE1B);
			case INTERRUPT::ON_INPUT_CAPTURE:
				 SET(TIMSK1, ICIE1);
			case INTERRUPT::ON_OVERFLOW:
				 SET(TIMSK1, TOIE1);
		}
	}
	template <Timer1::INTERRUPT interrupt>
	inline void Timer1::disable(void)
	{
		switch (interrupt)
		{
			case INTERRUPT::ON_OUTPUT_COMPARE_MATCH_A:
				 CLEAR(TIMSK1, OCIE1A);
			case INTERRUPT::ON_OUTPUT_COMPARE_MATCH_B:
				 CLEAR(TIMSK1, OCIE1B);
			case INTERRUPT::ON_INPUT_CAPTURE:
				 CLEAR(TIMSK1, ICIE1);
			case INTERRUPT::ON_OVERFLOW:
				 CLEAR(TIMSK1, TOIE1);
		}
	}
	inline void Timer1::  pause(void) { clock_source(CLOCK_SOURCE::NO   ); }
	inline void Timer1::unpause(void) { clock_source(m_spec.clock_source); }
	//
	inline void Timer1::force_output_compare_A(void) { SET(TCCR1B, FOC1A); }
	inline void Timer1::force_output_compare_B(void) { SET(TCCR1B, FOC1B); }
	//
	template <Timer1::INTERRUPT interrupt>
	inline void Timer1::call(void)
	{
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

	extern Timer1 timer1;
}

#endif
