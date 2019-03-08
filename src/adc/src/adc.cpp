#include <avr/io.h>

#include <standard/standard.h>

#include "adc.h"

namespace Adc
{
	void begin(const INIT* init)
	{
#if defined(__AVR_ATmega48P__) || defined(__AVR_ATmega88P__) || defined(__AVR_ATmega168P__) || defined(__AVR_ATmega328P__)
		PRR    &= ~(1 << PRADC);
#endif
		uint8_t prescaler_select = static_cast<uint8_t>(init->prescaler_select);
		write_bit({&ADCSRA, ADPS0}, prescaler_select & 0b001);
		write_bit({&ADCSRA, ADPS1}, prescaler_select & 0b010);
		write_bit({&ADCSRA, ADPS2}, prescaler_select & 0b100);

		uint8_t vref = static_cast<uint8_t>(init->vref);
		write_bit({&ADMUX, REFS0}, vref & 0b01);
		write_bit({&ADMUX, REFS1}, vref & 0b10);

		ADCSRA |=  (1 << ADEN) | (1 << ADIE);
	}

	void begin_simple()
	{
#if defined(__AVR_ATmega48P__) || defined(__AVR_ATmega88P__) || defined(__AVR_ATmega168P__) || defined(__AVR_ATmega328P__)
		PRR    &= ~(1 << PRADC);
#endif
		uint8_t prescaler_select = static_cast<uint8_t>(Adc::PRESCALER_SELECT::S128);
		write_bit({&ADCSRA, ADPS0}, prescaler_select & 0b001);
		write_bit({&ADCSRA, ADPS1}, prescaler_select & 0b010);
		write_bit({&ADCSRA, ADPS2}, prescaler_select & 0b100);

		uint8_t vref = static_cast<uint8_t>(Adc::VREF::AVCC);
		write_bit({&ADMUX, REFS0}, vref & 0b01);
		write_bit({&ADMUX, REFS1}, vref & 0b10);

		ADCSRA |=  (1 << ADEN) | (1 << ADIE);
	}

	void select_channel(uint8_t channel)
	{
		ADMUX = (ADMUX & 0b11100000) | channel;
	}

	void start_conversion()
	{
		ADCSRA |= (1 << ADSC);
	}
}
