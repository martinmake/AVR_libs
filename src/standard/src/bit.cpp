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

void Bit::set()
{
	m_reg |=  (1 << m_index);
}

void Bit::clear()
{
	m_reg &= ~(1 << m_index);
}

void Bit::write(uint8_t val)
{
	if (val)
		set();
	else
		clear();
}

uint8_t Bit::read() const
{
	return m_reg & (1 << m_index);
}

Bit Bit::operator-(int i) const
{
	return {*(&m_reg - i), m_index};
}

#endif
