#ifndef _TIMER_TIMER2_H_
#define _TIMER_TIMER2_H_

#include "timer/base.h"

namespace Timer
{
	class Timer2 : public Timer::Base<uint8_t>
	{
		public: // CONSTRUCTORS
			Timer2(void) = default;
			Timer2(const Spec& spec);
		public: // DESTRUCTOR
			~Timer2(void) = default;

		public: // GETTERS
			uint8_t count                 (void) const override;
			uint8_t output_compare_value_A(void) const override;
			uint8_t output_compare_value_B(void) const override;
			uint8_t input_capture_value   (void) const override = delete;
		public: // SETTERS
			void count                 (uint8_t new_count                 ) override;
			void output_compare_value_A(uint8_t new_output_compare_value_A) override;
			void output_compare_value_B(uint8_t new_output_compare_value_B) override;
			void input_capture_value   (uint8_t new_input_capture_value   ) override = delete;
			//
			void pin_action_on_output_compare_match_A(PIN_ACTION_ON_OUTPUT_COMPARE_MATCH new_pin_action_on_output_compare_match_A) override;
			void pin_action_on_output_compare_match_B(PIN_ACTION_ON_OUTPUT_COMPARE_MATCH new_pin_action_on_output_compare_match_B) override;
			void input_capture(INPUT_CAPTURE new_input_capture) override;
			void clock_source(CLOCK_SOURCE new_clock_source) override;
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
	inline uint8_t Timer2::count                 (void) const { return TCNT0; }
	inline uint8_t Timer2::output_compare_value_A(void) const { return OCR0A; }
	inline uint8_t Timer2::output_compare_value_B(void) const { return OCR0B; }
	// SETTERS
	inline void Timer2::count                 (uint8_t new_count                 ) { TCNT0 = new_count;                  }
	inline void Timer2::output_compare_value_A(uint8_t new_output_compare_value_A) { OCR0A = new_output_compare_value_A; }
	inline void Timer2::output_compare_value_B(uint8_t new_output_compare_value_B) { OCR0B = new_output_compare_value_B; }
	//
	inline void Timer2::pin_action_on_output_compare_match_A(
			PIN_ACTION_ON_OUTPUT_COMPARE_MATCH new_pin_action_on_output_compare_match_A)
	{
		switch (new_pin_action_on_output_compare_match_A)
		{
			case PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::PASS:
				CLEAR(TCCR2A, COM2A0);
				CLEAR(TCCR2A, COM2A1);
				break;
			case PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::TOGGLE:
				SET  (TCCR2A, COM2A0);
				CLEAR(TCCR2A, COM2A1);
				break;
			case PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::CLEAR:
				CLEAR(TCCR2A, COM2A0);
				SET  (TCCR2A, COM2A1);
				break;
			case PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::SET:
				SET  (TCCR2A, COM2A0);
				SET  (TCCR2A, COM2A1);
				break;
			default: assert(false && "INVALID VALUE");
		}
	}
	inline void Timer2::pin_action_on_output_compare_match_B(
		PIN_ACTION_ON_OUTPUT_COMPARE_MATCH new_pin_action_on_output_compare_match_B)
	{
		switch (new_pin_action_on_output_compare_match_B)
		{
			case PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::PASS:
				CLEAR(TCCR2A, COM2B0);
				CLEAR(TCCR2A, COM2B1);
				break;
			case PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::TOGGLE:
				SET  (TCCR2A, COM2B0);
				CLEAR(TCCR2A, COM2B1);
				break;
			case PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::CLEAR:
				CLEAR(TCCR2A, COM2B0);
				SET  (TCCR2A, COM2B1);
				break;
			case PIN_ACTION_ON_OUTPUT_COMPARE_MATCH::SET:
				SET  (TCCR2A, COM2B0);
				SET  (TCCR2A, COM2B1);
				break;
			default: assert(false && "INVALID VALUE");
		}
	}
	inline void Timer2::input_capture(INPUT_CAPTURE new_input_capture)
	{
		assert(new_input_capture == INPUT_CAPTURE::DISABLED);
	}
	inline void Timer2::clock_source(CLOCK_SOURCE new_clock_source)
	{
		m_spec.clock_source = new_clock_source;

		switch (new_clock_source)
		{
			case CLOCK_SOURCE::NO:
				CLEAR(TCCR2B, CS20);
				CLEAR(TCCR2B, CS21);
				CLEAR(TCCR2B, CS22);
				break;
			case CLOCK_SOURCE::IO_CLK_OVER_1:
				SET  (TCCR2B, CS20);
				CLEAR(TCCR2B, CS21);
				CLEAR(TCCR2B, CS22);
				break;
			case CLOCK_SOURCE::IO_CLK_OVER_8:
				CLEAR(TCCR2B, CS20);
				SET  (TCCR2B, CS21);
				CLEAR(TCCR2B, CS22);
				break;
			case CLOCK_SOURCE::IO_CLK_OVER_32:
				SET  (TCCR2B, CS20);
				SET  (TCCR2B, CS21);
				CLEAR(TCCR2B, CS22);
				break;
			case CLOCK_SOURCE::IO_CLK_OVER_64:
				CLEAR(TCCR2B, CS20);
				CLEAR(TCCR2B, CS21);
				SET  (TCCR2B, CS22);
				break;
			case CLOCK_SOURCE::IO_CLK_OVER_128:
				SET  (TCCR2B, CS20);
				CLEAR(TCCR2B, CS21);
				SET  (TCCR2B, CS22);
				break;
			case CLOCK_SOURCE::IO_CLK_OVER_256:
				CLEAR(TCCR2B, CS20);
				SET  (TCCR2B, CS21);
				SET  (TCCR2B, CS22);
				break;
			case CLOCK_SOURCE::IO_CLK_OVER_1024:
				SET  (TCCR2B, CS20);
				SET  (TCCR2B, CS21);
				SET  (TCCR2B, CS22);
				break;
			default: assert(false && "INVALID VALUE");
		}
	}
	//
	inline void Timer2::on_output_compare_match_A(bool enable)
	{
		if (enable) SET  (TIMSK2, OCIE2A);
		else        CLEAR(TIMSK2, OCIE2A);
	}
	inline void Timer2::on_output_compare_match_B(bool enable)
	{
		if (enable) SET  (TIMSK2, OCIE2B);
		else        CLEAR(TIMSK2, OCIE2B);
	}
	inline void Timer2::on_overflow(bool enable)
	{
		if (enable) SET  (TIMSK2, TOIE2);
		else        CLEAR(TIMSK2, TOIE2);
	}

	// METHODS
	inline void Timer2::update_waveform_generation(void)
	{
		switch (m_spec.mode)
		{
			case MODE::NORMAL: switch (m_spec.top)
			{
				case TOP::MAX:
					CLEAR(TCCR2A, WGM20);
					CLEAR(TCCR2A, WGM21);
					CLEAR(TCCR2A, WGM22);
					break;
				default: assert(false && "INVALID VALUE");
			} break;
			case MODE::CTC: switch (m_spec.top)
			{
				case TOP::OUTPUT_COMPARE_VALUE_A:
					CLEAR(TCCR2A, WGM20);
					SET  (TCCR2A, WGM21);
					CLEAR(TCCR2A, WGM22);
					break;
				default: assert(false && "INVALID VALUE");
			} break;
			case MODE::FAST_PWM: switch (m_spec.top)
			{
				case TOP::MAX:
					SET  (TCCR2A, WGM20);
					SET  (TCCR2A, WGM21);
					CLEAR(TCCR2A, WGM22);
					break;
				case TOP::OUTPUT_COMPARE_VALUE_A:
					SET  (TCCR2A, WGM20);
					SET  (TCCR2A, WGM21);
					SET  (TCCR2A, WGM22);
					break;
				default: assert(false && "INVALID VALUE");
			} break;
			case MODE::PHASE_CORRECT_PWM: switch (m_spec.top)
			{
				case TOP::MAX:
					SET  (TCCR2A, WGM20);
					CLEAR(TCCR2A, WGM21);
					CLEAR(TCCR2A, WGM22);
					break;
				case TOP::OUTPUT_COMPARE_VALUE_A:
					SET  (TCCR2A, WGM20);
					CLEAR(TCCR2A, WGM21);
					SET  (TCCR2A, WGM22);
					break;
				default: assert(false && "INVALID VALUE");
			} break;
			default: assert(false && "INVALID VALUE");
		}
		default: assert(false && "INVALID VALUE");
	}
	//
	inline void Timer2::force_output_compare_A(void) { SET(TCCR2B, FOC2A); }
	inline void Timer2::force_output_compare_B(void) { SET(TCCR2B, FOC2B); }

	extern Timer2 timer2;
}

#endif
