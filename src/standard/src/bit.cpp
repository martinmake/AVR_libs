#ifndef _STANDARD_BIT_H_
#define _STANDARD_BIT_H_

#include "standard.h"

Bit::Bit(volatile uint8_t& reg, uint8_t index)
	: m_reg(reg), m_index(index)
{
}

Bit::~Bit()
{
}

Bit Bit::operator-(int i) const
{
	return {*(&m_reg - i), m_index};
}

#endif
