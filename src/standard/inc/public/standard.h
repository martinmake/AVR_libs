#ifndef _STANDARD_STANDARD_H_
#define _STANDARD_STANDARD_H_

#include <stddef.h>
#include <inttypes.h>

typedef enum {
	BIN =  2,
	OCT =  8,
	DEC = 10,
	HEX = 16
} BASE;

class Bit
{
	public:
		volatile uint8_t& m_reg;
		uint8_t m_index;

	public:
		Bit(volatile uint8_t& reg, uint8_t index);
		~Bit();

		inline void set()              { m_reg |=  (1 << m_index);      }
		inline void clear()            { m_reg &= ~(1 << m_index);      }
		inline void write(uint8_t val) { val ? set() : clear();         }
		inline uint8_t read() const    { return m_reg & (1 << m_index); }

		Bit operator-(int i) const;
};

class Pin
{
	public:
		Bit port;
		Bit dd;
		Bit pin;

	public:
		Pin(const Bit& port_reg);
		~Pin();
};

extern void* operator new(size_t size);
extern void* operator new[](size_t size);
extern void  operator delete(void* ptr);
extern void  operator delete[](void* ptr);

#endif
