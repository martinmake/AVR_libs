#include "vl53l0x/base.h"

namespace Vl53l0x
{
	Base::Base(uint8_t initial_address)
		: m_address(initial_address)
	{
	}

	Base::~Base(void)
	{
	}

	void Base::init(void)
	{
	}
}
