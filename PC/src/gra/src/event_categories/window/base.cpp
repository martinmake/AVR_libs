#include <utility>

#include "logging.h"

#include "gra/event_categories/window/base.h"

namespace Gra
{
	namespace EventCategory
	{
		namespace Window
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
