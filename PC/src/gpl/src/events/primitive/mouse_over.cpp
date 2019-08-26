#include "gpl/events/primitive/mouse_over.h"

namespace Gpl
{
	namespace Event
	{
		namespace Primitive
		{
			MouseOver::MouseOver(Primitive::Base& instance)
				: Base(instance)
			{
			}
			MouseOver::~MouseOver(void)
			{
			}

			void MouseOver::copy(const MouseOver& other)
			{
				Event::Primitive::Base::copy(other);
			}
			void MouseOver::move(MouseOver&& other)
			{
				Event::Primitive::Base::move(std::move(other));
			}
		}
	}
}
