#include "bit.h"

Bit::Bit(volatile uint8_t& reg, uint8_t index)
	: m_reg(reg), m_index(index)
{
}

Bit::Bit(const Bit& bit, int8_t offset)
	: m_reg(*(&bit.m_reg + offset)),
	  m_index(bit.m_index)
{
}

Bit::~Bit()
{
}
