#ifndef _STANDARD_BIT_H_
#define _STANDARD_BIT_H_

#include "standard.h"

Bit::Bit(volatile uint8_t* addr, uint8_t index)
	: m_addr(addr), m_index(index)
{
}

Bit::~Bit()
{
}

void Bit::set() const
{
	*m_addr |=  (1 << m_index);
}

void Bit::clear() const
{
	*m_addr &= ~(1 << m_index);
}

void Bit::write(uint8_t val) const
{
	if (val)
		set();
	else
		clear();
}

uint8_t Bit::read() const
{
	return *m_addr & (1 << m_index);
}

Bit Bit::operator-(int i) const
{
	return {m_addr - i, m_index};
}

#endif
