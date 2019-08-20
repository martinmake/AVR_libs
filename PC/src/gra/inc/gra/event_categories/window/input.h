#ifndef _GRA_EVENT_CATEGORY_WINDOW_INPUT_H_
#define _GRA_EVENT_CATEGORY_WINDOW_INPUT_H_

#include <sml/sml.h>

#include "gra/core.h"
#include "gra/event_categories/window/base.h"

namespace Gra
{
	namespace EventCategory
	{
		namespace Window
		{
			class Input : public EventCategory::Window::Base
			{
				public:
					enum class Action
					{
						PRESS   = GLFW_PRESS,
						RELEASE = GLFW_RELEASE,
					};
					enum class Mod
					{
						SHIFT     = GLFW_MOD_SHIFT,
						CONTROL   = GLFW_MOD_CONTROL,
						ALT       = GLFW_MOD_ALT,
						SUPER     = GLFW_MOD_SUPER,
						CAPS_LOCK = GLFW_MOD_CAPS_LOCK,
						NUM_LOCK  = GLFW_MOD_NUM_LOCK,
					};

				DECLARATION_MANDATORY_INTERFACE(Input)
			};
			inline bool operator&(Input::Mod lhs, Input::Mod rhs)
			{
				return static_cast<uint8_t>(lhs) & static_cast<uint8_t>(rhs);
			}

			DEFINITION_MANDATORY_INTERFACE(Input, )
		}
	}
}

#endif
