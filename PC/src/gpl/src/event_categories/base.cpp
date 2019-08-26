#include <utility>

#include "gpl/event_categories/base.h"

namespace Gpl
{
	namespace EventCategory
	{
		Base::Base(void)
		{
		}

		Base::~Base(void)
		{
		}

		void Base::copy(const Base& other)
		{
			Esl::EventCategory::Base::copy(other);
		}
		void Base::move(Base&& other)
		{
			Esl::EventCategory::Base::move(std::move(other));
		}
	}
}
