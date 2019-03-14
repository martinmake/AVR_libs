#include "spi.h"

namespace Spi
{
	void begin(const INIT* init)
	{
		init->sck_ddb.set();
		init->miso_ddb.clear();
		init->mosi_ddb.set();

		uint8_t clock_rate_select = static_cast<uint8_t>(init->clock_rate_select);
		Bit(&SPCR, SPR0 ).write(clock_rate_select & 0b001);
		Bit(&SPCR, SPR1 ).write(clock_rate_select & 0b010);
		Bit(&SPSR, SPI2X).write(clock_rate_select & 0b100);

		SPCR |= (1 << SPE) | (1 << MSTR);
	}

	uint8_t send(uint8_t data)
	{
		SPDR = data;

		while (!(SPSR & (1 << SPIF))) {}

		return SPDR;
	}

	Slave::Slave(Pin ss)
		: m_ss(ss)
	{
		ss.dd.set();
		this->unselect();
	}

	Slave::~Slave()
	{
		this->unselect();
	}

	void Slave::select()
	{
		m_ss.port.clear();
	}

	void Slave::unselect()
	{
		m_ss.port.set();
	}
}
