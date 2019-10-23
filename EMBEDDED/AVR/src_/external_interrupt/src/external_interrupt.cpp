#include "external_interrupt.h"

ExternalInterrupt::on_interrupt_func ExternalInterrupt::on_interrupt0 = nullptr;
ExternalInterrupt::on_interrupt_func ExternalInterrupt::on_interrupt1 = nullptr;

ExternalInterrupt::ExternalInterrupt(const Spec& spec)
{
	init(spec);
}

void ExternalInterrupt::init(const Spec& spec)
{
	switch (spec.interrupt)
	{
		case INTERRUPT0: switch (spec.sense)
		{
			case SENSE::LOW_LEVEL:
				CLEAR(EICRA, ISC00);
				CLEAR(EICRA, ISC01);
				break;
			case SENSE::LOGICAL_CHANGE:
				SET  (EICRA, ISC00);
				CLEAR(EICRA, ISC01);
				break;
			case SENSE::FALLING_EDGE:
				CLEAR(EICRA, ISC00);
				SET  (EICRA, ISC01);
				break;
			case SENSE::RISING_EDGE:
				SET  (EICRA, ISC00);
				SET  (EICRA, ISC01);
				break;
		} on_interrupt0 = spec.on_interrupt; SET(EIMSK, INT0); break;
		case INTERRUPT1: switch (spec.sense)
		{
			case SENSE::LOW_LEVEL:
				CLEAR(EICRA, ISC10);
				CLEAR(EICRA, ISC11);
				break;
			case SENSE::LOGICAL_CHANGE:
				SET  (EICRA, ISC10);
				CLEAR(EICRA, ISC11);
				break;
			case SENSE::FALLING_EDGE:
				CLEAR(EICRA, ISC10);
				SET  (EICRA, ISC11);
				break;
			case SENSE::RISING_EDGE:
				SET  (EICRA, ISC10);
				SET  (EICRA, ISC11);
				break;
		} on_interrupt1 = spec.on_interrupt; SET(EIMSK, INT1); break;
	}
}

ISR(INT0_vect) { ExternalInterrupt::on_interrupt0(); }
ISR(INT1_vect) { ExternalInterrupt::on_interrupt1(); }
