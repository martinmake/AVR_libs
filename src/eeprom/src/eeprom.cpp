#include "eeprom.h"

namespace Eeprom
{
	void write(uint16_t address, uint8_t data)
	{
		while (EECR & (1 << EEPE))
			;

		EEAR = address;
		EEDR = data;

		EECR |= (1 << EEMPE);
		EECR |= (1 << EEPE);
	}

	uint8_t read(uint16_t address)
	{
		while (EECR & (1 << EEPE))
			;

		EEAR = address;

		EECR |= (1 << EERE);

		return EEDR;
	}
}
