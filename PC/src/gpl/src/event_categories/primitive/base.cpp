#include <utility>

#include "gpl/event_categories/primitive/base.h"

namespace Gpl
{
	namespace EventCategory
	{
		namespace Primitive
		{
			Base::Base(void)
			{
			}

			Base::~Base(void)
			{
			}

			void Base::copy(const Base& other)
			{
				EventCategory::Base::copy(other);
			}
			void Base::move(Base&& other)
			{
				EventCategory::Base::move(std::move(other));
			}
		}
	}
}
