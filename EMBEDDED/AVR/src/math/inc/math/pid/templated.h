#ifndef _MATH_PID_TEMPLATED_H_
#define _MATH_PID_TEMPLATED_H_

#include <util.h>

namespace Pid
{
	template <
		typename input_t,
		typename intermediate_t,
		typename output_t,
		intermediate_t kp,
		intermediate_t ki,
		intermediate_t kd,
		intermediate_t integral_limit,
		intermediate_t factor>
	class Templated
	{
		public: // CONSTRUCTORS
			Templated(void) = default;
		public: // DESTRUCTOR
			~Templated(void) = default;

		public: // METHODS
			void init (void);
			void reset(void);

		public: // OPERATORS
			output_t operator()(input_t error);

		private:
			intermediate_t m_error_sum  = 0;
			intermediate_t m_last_error = 0;
	};
	// METHODS
	template <
		typename input_t,
		typename intermediate_t,
		typename output_t,
		intermediate_t kp,
		intermediate_t ki,
		intermediate_t kd,
		intermediate_t integral_limit,
		intermediate_t factor>
	void Templated<
		input_t,
		intermediate_t,
		output_t,
	        kp,
		ki,
		kd,
		integral_limit,
		factor>
	::init(void)
	{
	}
	template <
		typename input_t,
		typename intermediate_t,
		typename output_t,
		intermediate_t kp,
		intermediate_t ki,
		intermediate_t kd,
		intermediate_t integral_limit,
		intermediate_t factor>
	void Templated<
		input_t,
		intermediate_t,
		output_t,
	        kp,
		ki,
		kd,
		integral_limit,
		factor>
	::reset(void)
	{
		m_error_sum  = 0;
		m_last_error = 0;
	}

	// OPERATORS
	template <
		typename input_t,
		typename intermediate_t,
		typename output_t,
		intermediate_t kp,
		intermediate_t ki,
		intermediate_t kd,
		intermediate_t integral_limit,
		intermediate_t factor>
	output_t Templated<
		input_t,
		intermediate_t,
		output_t,
	        kp,
		ki,
		kd,
		integral_limit,
		factor>
	::operator()(input_t error)
	{
		m_error_sum = safe_add(m_error_sum, error);
		m_error_sum = clamp<intermediate_t,
			-integral_limit, integral_limit>(m_error_sum);

		intermediate_t error_change = error - m_last_error;
		m_last_error = error;

		output_t controll = safe_cast<output_t>(
			(kp * error +
			 ki * m_error_sum +
			 kd * error_change) / factor);

		return controll;
	}
}

#endif
