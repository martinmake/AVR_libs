#ifndef _ADC_ADC_H_
#define _ADC_ADC_H_

namespace Adc
{
	typedef enum class prescaler_select : uint8_t {
		S2   = 0b000,
		S4   = 0b010,
		S8   = 0b011,
		S16  = 0b100,
		S32  = 0b101,
		S64  = 0b110,
		S128 = 0b111
	} PRESCALER_SELECT;

	typedef enum class vref : uint8_t {
		AREF = 0b00,
		AVCC = 0b01,
		IREF = 0b11
	} VREF;

	typedef struct init {
		PRESCALER_SELECT prescaler_select;
		VREF vref;
	} INIT;

	void begin(const INIT* init);
	void begin_simple();
	void select_channel(uint8_t channel);
	void start_conversion();
}

#endif
