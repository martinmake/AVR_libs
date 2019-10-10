#include <utility>

#include "gra/inputs/base.h"

namespace Gra
{
	namespace Input
	{
		Base::Base(void)
		{
		}

		Base::~Base(void)
		{
		}

		void Base::copy(const Base& other)
		{
			(void) other;
		}
		void Base::move(Base&& other)
		{
			(void) other;
		}
	}
}
