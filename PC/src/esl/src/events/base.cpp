#include <utility>

#include "esl/events/base.h"

namespace Esl
{
	namespace Event
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
