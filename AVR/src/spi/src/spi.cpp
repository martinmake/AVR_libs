#include "spi.h"

Spi::Spi(Init* init)
{
	init->sck_ddb  = 1;
	init->miso_ddb = 0;
	init->mosi_ddb = 1;

	switch (init->clock_rate_select) {
		case ClockRateSelect::S2:
			Bit(SPCR, SPR0 ) = 0;
			Bit(SPCR, SPR1 ) = 0;
			Bit(SPSR, SPI2X) = 1;
			break;
		case ClockRateSelect::S4:
			Bit(SPCR, SPR0 ) = 0;
			Bit(SPCR, SPR1 ) = 0;
			Bit(SPSR, SPI2X) = 0;
			break;
		case ClockRateSelect::S8:
			Bit(SPCR, SPR0 ) = 1;
			Bit(SPCR, SPR1 ) = 0;
			Bit(SPSR, SPI2X) = 1;
			break;
		case ClockRateSelect::S16:
			Bit(SPCR, SPR0 ) = 1;
			Bit(SPCR, SPR1 ) = 0;
			Bit(SPSR, SPI2X) = 0;
			break;
		case ClockRateSelect::S32:
			Bit(SPCR, SPR0 ) = 0;
			Bit(SPCR, SPR1 ) = 1;
			Bit(SPSR, SPI2X) = 1;
			break;
		case ClockRateSelect::S64:
			Bit(SPCR, SPR0 ) = 0;
			Bit(SPCR, SPR1 ) = 1;
			Bit(SPSR, SPI2X) = 0;
			break;
		case ClockRateSelect::S128:
			Bit(SPCR, SPR0 ) = 1;
			Bit(SPCR, SPR1 ) = 1;
			Bit(SPSR, SPI2X) = 0;
			break;
	}

	SPCR |= (1 << SPE) | (1 << MSTR);
}
