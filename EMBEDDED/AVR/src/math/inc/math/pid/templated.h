#ifndef _MATH_PID_H_
#define _MATH_PID_H_

#include <util.h>

template <typename input_t, typename intermediate_t, typename output_t, input_t kp, input_t ki, input_t kd, input_t limit>
class Pid
{
	public: // CONSTRUCTORS
		Pid(void);
	public: // DESTRUCTOR
		~Pid(void);

	public: // METHODS
		void init (void);
		void reset(void);

	public: // OPERATORS
		output_t operator()(input_t desired_output, input_t actual_output);

	private:
		intermediate_t m_error_sum  = 0;
		intermediate_t m_last_error = 0;
};

// CONSTRUCTORS
template <typename input_t, typename intermediate_t, typename output_t, input_t kp, input_t ki, input_t kd, input_t limit>
Pid<input_t, intermediate_t, output_t, kp, ki, kd, limit>::Pid(void)
{
}
// DESTRUCTOR
template <typename input_t, typename intermediate_t, typename output_t, input_t kp, input_t ki, input_t kd, input_t limit>
Pid<input_t, intermediate_t, output_t, kp, ki, kd, limit>::~Pid(void)
{
}

// METHODS
template <typename input_t, typename intermediate_t, typename output_t, input_t kp, input_t ki, input_t kd, input_t limit>
void Pid<input_t, intermediate_t, output_t, kp, ki, kd, limit>::init(void)
{
}
template <typename input_t, typename intermediate_t, typename output_t, input_t kp, input_t ki, input_t kd, input_t limit>
void Pid<input_t, intermediate_t, output_t, kp, ki, kd, limit>::reset(void)
{
	m_error_sum  = 0;
	m_last_error = 0;
}

// OPERATORS
template <typename input_t, typename intermediate_t, typename output_t, input_t kp, input_t ki, input_t kd, input_t limit>
output_t Pid<input_t, intermediate_t, output_t, kp, ki, kd, limit>::operator()(input_t desired_output, input_t actual_output)
{
	input_t error = desired_output - actual_output;
	m_error_sum = safe_add(m_error_sum, error);
	if (m_error_sum >  limit) m_error_sum =  limit;
	if (m_error_sum < -limit) m_error_sum = -limit;

	intermediate_t error_change = error - m_last_error;
	m_last_error = error;

	output_t controll = clamp<intermediate_t, 0, limit>((intermediate_t) kp*error + ki*m_error_sum + kd*error_change);
	if (controll >  limit) controll =  limit;
	if (controll < -limit) controll = -limit;

	return controll;
}

#endif
