#ifndef _MULTITASKING_PREEMPTIVE_LED_H_
#define _MULTITASKING_PREEMPTIVE_LED_H_

#include <standard/standard.h>

class Led
{
	private:
		Bit m_port;
		bool m_inverted;

	public:
		Led(Bit led_port, bool inverted = false);
		~Led();

		bool operator=(bool state);
};

extern Led led_err;

#endif
