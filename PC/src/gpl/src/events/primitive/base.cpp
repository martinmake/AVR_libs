#include <utility>

#include "gpl/events/primitive/base.h"

namespace Gpl
{
	namespace Event
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
				Event::Base::copy(other);
			}
			void Base::move(Base&& other)
			{
				Event::Base::move(std::move(other));
			}
		}
	}
}
