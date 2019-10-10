#include <utility>

#include "gpl/events/base.h"

namespace Gpl
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
			Esl::Event::Base::copy(other);
		}
		void Base::move(Base&& other)
		{
			Esl::Event::Base::move(std::move(other));
		}
	}
}
