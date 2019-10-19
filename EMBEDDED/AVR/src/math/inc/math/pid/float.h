#ifndef _MATH_PID_FLOAT_H_
#define _MATH_PID_FLOAT_H_

#include <util.h>

namespace Pid
{
	class Float
	{
		public: // CONSTRUCTORS
			Float(void) = default;
			Float(float kp, float ki, float kd, float limit);
		public: // DESTRUCTOR
			~Float(void) = default;

		public: // METHODS
			void init (float kp, float ki, float kd, float limit);
			void reset(void);

		public: // OPERATORS
			float operator()(float error);

		private:
			float m_kp;
			float m_ki;
			float m_kd;
			float m_limit;

			float m_error_sum;
			float m_last_error;
	};

	// CONSTRUCTORS
	Float::Float(float kp, float ki, float kd, float limit)
	{
		init(kp, ki, kd, limit);
	}

	// METHODS
	void Float::init(float kp, float ki, float kd, float limit)
	{
		reset();

		m_kp = kp;
		m_ki = ki;
		m_kd = kd;
		m_limit = limit;
	}
	void Float::reset(void)
	{
		m_error_sum  = 0;
		m_last_error = 0;
	}

	// OPERATORS
	float Float::operator()(float error)
	{
		m_error_sum = safe_add(m_error_sum, error);
		m_error_sum = clamp(-m_limit, m_limit, m_error_sum);

		float error_change = error - m_last_error;
		m_last_error = error;

		float control =
			m_kp * error +
			m_ki * m_error_sum +
			m_kd * error_change;

		return control;
	}
}

#endif
