#ifndef _TIMER_TIMER1_H_
#define _TIMER_TIMER1_H_

#include <util.h>

#include "core/timer/base.h"

namespace Core
{
	namespace Timer
	{
		class Timer1 : public Timer::Base<uint16_t>
		{
			public: // CONSTRUCTORS
				Timer1(void) = default;
				Timer1(const Spec& spec);
			public: // DESTRUCTOR
				~Timer1(void) = default;

			public: // GETTERS
				uint16_t count                 (void) const override;
				uint16_t output_compare_value_A(void) const override;
				uint16_t output_compare_value_B(void) const override;
				uint16_t input_capture_value   (void) const override;
			public: // SETTERS
				void count                 (uint16_t new_count                 ) override;
				void output_compare_value_A(uint16_t new_output_compare_value_A) override;
				void output_compare_value_B(uint16_t new_output_compare_value_B) override;
				void input_capture_value   (uint16_t new_input_capture_value   ) override;
				//
				void pin_action_on_output_compare_match_A(TIMER::PIN_ACTION_ON_OUTPUT_COMPARE_MATCH new_pin_action_on_output_compare_match_A) override;
				void pin_action_on_output_compare_match_B(TIMER::PIN_ACTION_ON_OUTPUT_COMPARE_MATCH new_pin_action_on_output_compare_match_B) override;
				void input_capture(TIMER::INPUT_CAPTURE new_input_capture) override;
				void clock_source(TIMER::CLOCK_SOURCE new_clock_source) override;
				//
				void on_output_compare_match_A(bool enable) override;
				void on_output_compare_match_B(bool enable) override;
				void on_overflow              (bool enable) override;
				void on_input_capture         (bool enable) override;

			public: // METHODS
				void force_output_compare_A(void) override;
				void force_output_compare_B(void) override;

			private:
				void update_waveform_generation(void) override;
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
			TIMER::PIN_ACTION_ON_OUTPUT_COMPARE_MATCH new_pin_action_on_output_compare_match_A)
		{
			m_spec.pin_action_on_output_compare_match_A = new_pin_action_on_output_compare_match_A;

			using namespace TIMER;
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
				default: assert(false && "INVALID VALUE");
			}
		}
		inline void Timer1::pin_action_on_output_compare_match_B(
			TIMER::PIN_ACTION_ON_OUTPUT_COMPARE_MATCH new_pin_action_on_output_compare_match_B)
		{
			m_spec.pin_action_on_output_compare_match_B = new_pin_action_on_output_compare_match_B;

			using namespace TIMER;
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
				default: assert(false && "INVALID VALUE");
			}
		}
		inline void Timer1::input_capture(TIMER::INPUT_CAPTURE new_input_capture)
		{
			m_spec.input_capture = new_input_capture;

			using namespace TIMER;
			switch (new_input_capture)
			{
				case INPUT_CAPTURE::DISABLED:
					break;
				case INPUT_CAPTURE::ON_FALLING_EDGE:
					CLEAR(TCCR1B, ICES1);
					break;
				case INPUT_CAPTURE::ON_RISING_EDGE:
					SET  (TCCR1B, ICES1);
					break;
				default: assert(false && "INVALID VALUE");
			}
		}
		inline void Timer1::clock_source(TIMER::CLOCK_SOURCE new_clock_source)
		{
			m_spec.clock_source = new_clock_source;

			using namespace TIMER;
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
					CLEAR(TCCR1B, CS10);
					SET  (TCCR1B, CS11);
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
				default: assert(false && "INVALID VALUE");
			}
		}
		inline void Timer1::on_output_compare_match_A(bool enable)
		{
			if (enable) SET  (TIMSK1, OCIE1A);
			else        CLEAR(TIMSK1, OCIE1A);
		}
		inline void Timer1::on_output_compare_match_B(bool enable)
		{
			if (enable) SET  (TIMSK1, OCIE1B);
			else        CLEAR(TIMSK1, OCIE1B);
		}
		inline void Timer1::on_overflow(bool enable)
		{
			if (enable) SET  (TIMSK1, TOIE1);
			else        CLEAR(TIMSK1, TOIE1);
		}
		inline void Timer1::on_input_capture(bool enable)
		{
			if (enable) SET  (TIMSK1, ICIE1);
			else        CLEAR(TIMSK1, ICIE1);
		}

		// METHODS
		inline void Timer1::update_waveform_generation(void)
		{
			using namespace TIMER;
			switch (m_spec.mode)
			{
				case MODE::NORMAL: switch (m_spec.top)
				{
					case TOP::MAX:
						CLEAR(TCCR1A, WGM10);
						CLEAR(TCCR1A, WGM11);
						CLEAR(TCCR1B, WGM12);
						CLEAR(TCCR1B, WGM13);
						break;
					default: assert(false && "INVALID VALUE");
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
					default: assert(false && "INVALID VALUE");
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
					default: assert(false && "INVALID VALUE");
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
					default: assert(false && "INVALID VALUE");
				} break;
				case MODE::PHASE_AND_FREQUENCY_CORRECT_PWM: switch (m_spec.top)
				{
					case TOP::OUTPUT_COMPARE_VALUE_A:
						SET  (TCCR1A, WGM10);
						CLEAR(TCCR1A, WGM11);
						CLEAR(TCCR1B, WGM12);
						SET  (TCCR1B, WGM13);
						break; case TOP::INPUT_CAPTURE_VALUE: CLEAR(TCCR1A, WGM10); CLEAR(TCCR1A, WGM11); CLEAR(TCCR1B, WGM12);
						SET  (TCCR1B, WGM13);
						break;
					default: assert(false && "INVALID VALUE");
				} break;
				default: assert(false && "INVALID VALUE");
			}
		}
		//
		inline void Timer1::force_output_compare_A(void) { SET(TCCR1B, FOC1A); }
		inline void Timer1::force_output_compare_B(void) { SET(TCCR1B, FOC1B); }
	}

	extern Timer::Timer1 timer1;
}

#endif
