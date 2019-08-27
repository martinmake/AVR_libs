#include <utility>

#include "logging.h"

#include "gra/inputs/window/base.h"

namespace Gra
{
	namespace Input
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
				Input::Base::copy(other);
			}
			void Base::move(Base&& other)
			{
				Input::Base::move(std::move(other));
			}
		}
	}
}
