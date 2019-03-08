#include <avr/io.h>

#include "spi.h"

namespace Spi
{
	bool begun = false;

	void begin(const INIT* init)
	{
		set_bit(init->sck_ddb);
		clear_bit(init->miso_ddb);
		set_bit(init->mosi_ddb);

		uint8_t clock_rate_select = static_cast<uint8_t>(init->clock_rate_select);
		SPCR |= (clock_rate_select & 0b001 << SPR0);
		SPCR |= (clock_rate_select & 0b010 << SPR1);
		SPSR |= (clock_rate_select & 0b100 << SPI2X);

		SPCR |= (1 << SPE) | (1 << MSTR);

		begun = true;
	}

	uint8_t send(uint8_t data)
	{
		SPDR = data;

		while (!(SPSR & (1 << SPIF)))
			;

		return SPDR;
	}

	Slave::Slave(BIT ss_pin, BIT ss_ddb)
		: m_ss(ss_pin)
	{
		set_bit(ss_ddb);
		this->unselect();
	}

	Slave::~Slave()
	{
		this->unselect();
	}

	void Slave::select()
	{
		clear_bit(m_ss);
	}

	void Slave::unselect()
	{
		set_bit(m_ss);
	}
}
