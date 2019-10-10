#include <avr/interrupt.h>

#include "adc.h"

Adc adc;

Adc::Adc()
{
}

Adc::Adc(const Init& init_struct)
{
	init(init_struct);
}

void Adc::init(const Init& init_struct)
{
#if defined(__AVR_ATmega48P__) || defined(__AVR_ATmega88P__) || defined(__AVR_ATmega168P__) || defined(__AVR_ATmega328P__)
	CLEAR(PRR, PRADC);
#endif
	CLEAR(ADMUX, ADLAR);
	switch (init_struct.prescaler)
	{
		case PRESCALER::X2:
			SET  (ADCSRA, ADPS0);
			CLEAR(ADCSRA, ADPS1);
			CLEAR(ADCSRA, ADPS2);
			break;
		case PRESCALER::X4:
			CLEAR(ADCSRA, ADPS0);
			SET  (ADCSRA, ADPS1);
			CLEAR(ADCSRA, ADPS2);
			break;
		case PRESCALER::X8:
			SET  (ADCSRA, ADPS0);
			SET  (ADCSRA, ADPS1);
			CLEAR(ADCSRA, ADPS2);
			break;
		case PRESCALER::X16:
			CLEAR(ADCSRA, ADPS0);
			CLEAR(ADCSRA, ADPS1);
			SET  (ADCSRA, ADPS2);
			break;
		case PRESCALER::X32:
			SET  (ADCSRA, ADPS0);
			CLEAR(ADCSRA, ADPS1);
			SET  (ADCSRA, ADPS2);
			break;
		case PRESCALER::X64:
			CLEAR(ADCSRA, ADPS0);
			SET  (ADCSRA, ADPS1);
			SET  (ADCSRA, ADPS2);
			break;
		case PRESCALER::X128:
			SET  (ADCSRA, ADPS0);
			SET  (ADCSRA, ADPS1);
			SET  (ADCSRA, ADPS2);
			break;
	}

	switch (init_struct.auto_trigger_source)
	{
		case AUTO_TRIGGER_SOURCE::FREE_RUNNING:
			CLEAR(ADCSRB, ADTS0);
			CLEAR(ADCSRB, ADTS1);
			CLEAR(ADCSRB, ADTS2);
			break;
		case AUTO_TRIGGER_SOURCE::EXTERNAL_INTERRUPT_REQUEST_0:
			CLEAR(ADCSRB, ADTS0);
			SET  (ADCSRB, ADTS1);
			CLEAR(ADCSRB, ADTS2);
			break;
		case AUTO_TRIGGER_SOURCE::TIM0_COMPARE_MATCH_A:
			SET  (ADCSRB, ADTS0);
			SET  (ADCSRB, ADTS1);
			CLEAR(ADCSRB, ADTS2);
			break;
		case AUTO_TRIGGER_SOURCE::TIM0_OVERFLOW:
			CLEAR(ADCSRB, ADTS0);
			CLEAR(ADCSRB, ADTS1);
			SET  (ADCSRB, ADTS2);
			break;
		case AUTO_TRIGGER_SOURCE::TIM1_COMPARE_MATCH_B:
			SET  (ADCSRB, ADTS0);
			CLEAR(ADCSRB, ADTS1);
			SET  (ADCSRB, ADTS2);
			break;
		case AUTO_TRIGGER_SOURCE::TIM1_OVERFLOW:
			CLEAR(ADCSRB, ADTS0);
			SET  (ADCSRB, ADTS1);
			SET  (ADCSRB, ADTS2);
			break;
		case AUTO_TRIGGER_SOURCE::TIM1_CAPTURE_EVENT:
			SET  (ADCSRB, ADTS0);
			SET  (ADCSRB, ADTS1);
			SET  (ADCSRB, ADTS2);
			break;
	}

	switch (init_struct.vref)
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

ISR(ADC_vect) { adc.value = ADC; if (adc.on_conversion) adc.on_conversion(); }
