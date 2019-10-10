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

			m_is_handled = other.m_is_handled;
		}
		void Base::move(Base&& other)
		{
			(void) other;

			m_is_handled = std::exchange(other.m_is_handled, true);
		}
	}
}
