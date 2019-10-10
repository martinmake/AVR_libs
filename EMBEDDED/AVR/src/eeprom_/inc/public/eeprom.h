#ifndef _EEPROM_EEPROM_H_
#define _EEPROM_EEPROM_H_

#include <avr/io.h>

#include <standard/standard.h>

class Eeprom
{
	private:
		uint16_t m_address;

	public:
		Eeprom();
		~Eeprom();

		inline uint16_t get_address() { return m_address; }
		inline Eeprom& operator=(uint16_t address) { m_address = address; return *this; }
		inline Eeprom& operator[](uint16_t address) { m_address = address; return *this; }

		inline Eeprom& operator++()    { ++m_address; return *this; }
		inline Eeprom  operator++(int) { m_address++; return *this; }
		inline Eeprom& operator--()    { --m_address; return *this; }
		inline Eeprom  operator--(int) { m_address--; return *this; }

		Eeprom& operator<<(uint8_t  data);
		Eeprom& operator>>(uint8_t& data);
};

extern Eeprom eeprom;

#endif
