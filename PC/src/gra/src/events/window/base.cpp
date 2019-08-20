#include <utility>

#include "logging.h"

#include "gra/events/window/base.h"

namespace Gra
{
	namespace Event
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
				Event::Base::copy(other);
			}
			void Base::move(Base&& other)
			{
				Event::Base::move(std::move(other));
			}
		}
	}
}
