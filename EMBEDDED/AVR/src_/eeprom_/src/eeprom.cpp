#include "eeprom.h"

Eeprom::Eeprom()
	: m_address(0)
{
}

Eeprom::~Eeprom()
{
}

Eeprom& Eeprom::operator<<(uint8_t data)
{
	while (EECR & (1 << EEPE))
		;

	EEAR = m_address;
	EEDR = data;

	EECR |= (1 << EEMPE);
	EECR |= (1 << EEPE);

	return *this;
}

Eeprom& Eeprom::operator>>(uint8_t& data)
{
	while (EECR & (1 << EEPE))
		;

	EEAR = m_address;

	EECR |= (1 << EERE);

	data = EEDR;

	return *this;
}
