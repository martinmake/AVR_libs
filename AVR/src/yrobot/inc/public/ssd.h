#ifndef _YROBOT_SSD_H_
#define _YROBOT_SSD_H_

#include <avr/interrupt.h>

#include <standard/util.h>

#include "connections.h"

namespace Ssd
{
	extern uint8_t seg1;
	extern uint8_t seg2;

	void begin();
	void set_segment(char c);
	void display_num(uint8_t byte, BASE base);
	void display_str(char* s, uint16_t shift_speed = 500);

	void set_bus(uint8_t mask);
}

#ifdef YROBOT_SSD_DEFAULT_TIMER0_COMP_ISR
ISR(TIMER0_COMP_vect)
{
	static uint8_t counter = 0;
	static uint8_t segment_select = 1;

	if (counter == 5) {
		switch (segment_select) {
			case 1:
				LED6DIG2.port.set();
				Ssd::set_segment(Ssd::c1);
				LED6DIG1.port.clear();
				segment_select = 2;
				break;
			case 2:
				LED6DIG1.port.set();
				Ssd::set_segment(Ssd::c2);
				LED6DIG2.port.clear();
				segment_select = 1;
				break;
		}

		counter = 0;
	} else
		counter++;
}
#endif

#endif
