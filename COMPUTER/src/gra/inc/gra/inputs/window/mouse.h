#ifndef _GRA_INPUT_WINDOW_MOUSE_H_
#define _GRA_INPUT_WINDOW_MOUSE_H_

#include <sml/sml.h>

#include "gra/core.h"
#include "gra/inputs/window/base.h"

namespace Gra
{
	namespace Input
	{
		namespace Window
		{
			class Mouse : public Input::Window::Base
			{
				public:
					Mouse(void);

				public:
					enum class Button
					{
						LEFT   = GLFW_MOUSE_BUTTON_LEFT,
						RIGHT  = GLFW_MOUSE_BUTTON_RIGHT,
						MIDDLE = GLFW_MOUSE_BUTTON_MIDDLE,
						FOUR   = GLFW_MOUSE_BUTTON_4,
						FIVE   = GLFW_MOUSE_BUTTON_5,
						SIX    = GLFW_MOUSE_BUTTON_6,
						SEVEN  = GLFW_MOUSE_BUTTON_7,
						EIGHT  = GLFW_MOUSE_BUTTON_8,
					};

				DECLARATION_MANDATORY(Mouse)
			};

			DEFINITION_MANDATORY(Mouse, )
		}
	}
}

#endif
