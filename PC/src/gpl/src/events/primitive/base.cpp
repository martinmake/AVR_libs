#include <utility>

#include "gpl/events/primitive/base.h"

namespace Gpl
{
	namespace Event
	{
		namespace Primitive
		{
			Base::Base(void* initial_instance)
				: m_instance(initial_instance)
			{
			}
			Base::Base(void)
			{
			}

			Base::~Base(void)
			{
			}

			void Base::copy(const Base& other)
			{
				Event::Base::copy(other);

				m_instance = other.m_instance;
			}
			void Base::move(Base&& other)
			{
				Event::Base::move(std::move(other));

				m_instance = std::exchange(other.m_instance, nullptr);
			}
		}
	}
}
