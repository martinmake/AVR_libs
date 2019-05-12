#include <avr/io.h>

#include <standard/standard.h>

#include "adc.h"

Adc::Adc(const Init* init)
{
#if defined(__AVR_ATmega48P__) || defined(__AVR_ATmega88P__) || defined(__AVR_ATmega168P__) || defined(__AVR_ATmega328P__)
	Bit(PRR, PRADC).clear();
#endif
	switch (init->prescaler_select) {
		case PrescalerSelect::S2:
			Bit(ADCSRA, ADPS0).clear();
			Bit(ADCSRA, ADPS1).clear();
			Bit(ADCSRA, ADPS2).clear();
			break;
		case PrescalerSelect::S4:
			Bit(ADCSRA, ADPS0).clear();
			Bit(ADCSRA, ADPS1).set();
			Bit(ADCSRA, ADPS2).clear();
			break;
		case PrescalerSelect::S8:
			Bit(ADCSRA, ADPS0).set();
			Bit(ADCSRA, ADPS1).set();
			Bit(ADCSRA, ADPS2).clear();
			break;
		case PrescalerSelect::S16:
			Bit(ADCSRA, ADPS0).clear();
			Bit(ADCSRA, ADPS1).clear();
			Bit(ADCSRA, ADPS2).set();
			break;
		case PrescalerSelect::S32:
			Bit(ADCSRA, ADPS0).set();
			Bit(ADCSRA, ADPS1).clear();
			Bit(ADCSRA, ADPS2).set();
			break;
		case PrescalerSelect::S64:
			Bit(ADCSRA, ADPS0).clear();
			Bit(ADCSRA, ADPS1).set();
			Bit(ADCSRA, ADPS2).set();
			break;
		case PrescalerSelect::S128:
			Bit(ADCSRA, ADPS0).set();
			Bit(ADCSRA, ADPS1).set();
			Bit(ADCSRA, ADPS2).set();
			break;
	}

	switch (init->vref) {
		case Vref::AREF:
			Bit(ADMUX, REFS0).clear();
			Bit(ADMUX, REFS1).clear();
			break;
		case Vref::AVCC:
			Bit(ADMUX, REFS0).set();
			Bit(ADMUX, REFS1).clear();
			break;
		case Vref::IREF:
			Bit(ADMUX, REFS0).set();
			Bit(ADMUX, REFS1).set();
			break;
	}

	ADCSRA |=  (1 << ADEN) | (1 << ADIE);
}

Adc::Adc()
{
	Init init;

	init.prescaler_select = PrescalerSelect::S2;
	init.vref             = Vref::AVCC;

	*this = Adc(&init);
}
