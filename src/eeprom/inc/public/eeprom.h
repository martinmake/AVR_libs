#ifndef _EEPROM_EEPROM_H_
#define _EEPROM_EEPROM_H_

#include <avr/io.h>

#include <standard/standard.h>

namespace Eeprom
{
	void write(uint16_t address, uint8_t data);
	uint8_t read(uint16_t address);
}

#endif
