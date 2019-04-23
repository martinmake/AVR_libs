#ifndef _STANDARD_BIT_H_
#define _STANDARD_BIT_H_

#include <inttypes.h>

class Bit
{
	public:
		volatile uint8_t& m_reg;
		         uint8_t  m_index;

	public:
		Bit(volatile uint8_t& reg, uint8_t index);
		Bit(const Bit& bit, int8_t offset);
		~Bit();

		inline void set()                  { m_reg |=  (1 << m_index);            }
		inline void clear()                { m_reg &= ~(1 << m_index);            }
		inline void write(uint8_t val)     { val ? set() : clear();               }
		inline Bit& operator=(uint8_t val) { val ? set() : clear(); return *this; }
		inline uint8_t read() const        { return m_reg & (1 << m_index);       }
};

#endif
