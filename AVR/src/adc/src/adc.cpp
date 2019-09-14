#include <avr/interrupt.h>

#include "adc/adc.h"

Adc::Adc(const Init* init)
{
#if defined(__AVR_ATmega48P__) || defined(__AVR_ATmega88P__) || defined(__AVR_ATmega168P__) || defined(__AVR_ATmega328P__)
	PRR &= ~BIT(PRADC);
#endif
	switch (init->prescaler_select)
	{
		case PRESCALER_SELECT::X2:
			CLEAR(ADCSRA, ADPS0);
			CLEAR(ADCSRA, ADPS1);
			CLEAR(ADCSRA, ADPS2);
			break;
		case PRESCALER_SELECT::X4:
			CLEAR(ADCSRA, ADPS0);
			SET  (ADCSRA, ADPS1);
			CLEAR(ADCSRA, ADPS2);
			break;
		case PRESCALER_SELECT::X8:
			SET  (ADCSRA, ADPS0);
			SET  (ADCSRA, ADPS1);
			CLEAR(ADCSRA, ADPS2);
			break;
		case PRESCALER_SELECT::X16:
			CLEAR(ADCSRA, ADPS0);
			CLEAR(ADCSRA, ADPS1);
			SET  (ADCSRA, ADPS2);
			break;
		case PRESCALER_SELECT::X32:
			SET  (ADCSRA, ADPS0);
			CLEAR(ADCSRA, ADPS1);
			SET  (ADCSRA, ADPS2);
			break;
		case PRESCALER_SELECT::X64:
			CLEAR(ADCSRA, ADPS0);
			SET  (ADCSRA, ADPS1);
			SET  (ADCSRA, ADPS2);
			break;
		case PRESCALER_SELECT::X128:
			SET  (ADCSRA, ADPS0);
			SET  (ADCSRA, ADPS1);
			SET  (ADCSRA, ADPS2);
			break;
	}

	switch (init->vref)
	{
		case VREF::AREF:
			CLEAR(ADMUX, REFS0);
			CLEAR(ADMUX, REFS1);
			break;
		case VREF::AVCC:
			SET  (ADMUX, REFS0);
			CLEAR(ADMUX, REFS1);
			break;
		case VREF::IREF:
			SET  (ADMUX, REFS0);
			SET  (ADMUX, REFS1);
			break;
	}

	SET(ADCSRA, ADEN);
}

Adc::Adc()
{
	Init init;

	init.prescaler_select = PRESCALER_SELECT::X2;
	init.vref             = VREF::AVCC;

	*this = Adc(&init);
}

ISR(ADC_vect)
{
	if (adc.keep_sampling())
	{
		adc.ISR_callback(ADC);
		adc.start_conversion();
	}
}
