#include "standard.h"

void set_bit(BIT bit)
{
	*(bit.addr) |=  (1 << bit.index);
}

void clear_bit(BIT bit)
{
	*(bit.addr) &= ~(1 << bit.index);
}

void write_bit(BIT bit, uint8_t val)
{
	if (val)
		set_bit(bit);
	else
		clear_bit(bit);
}

uint8_t read_bit(BIT bit)
{
	return *(bit.addr) & (1 << bit.index);
}
