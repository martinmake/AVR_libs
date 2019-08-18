#include "esl/event_categories/base.h"

namespace Esl
{
	namespace EventCategories
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
