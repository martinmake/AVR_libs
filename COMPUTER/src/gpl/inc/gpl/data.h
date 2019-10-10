#ifndef _GPL_DATA_H_
#define _GPL_DATA_H_

#include <gra/graphics_objects/program.h>

#include "gpl/core.h"
#include "gpl/events/all.h"

namespace Gpl
{
	namespace Data
	{
		struct Draw
		{
			Gra::Math::vec2<unsigned int> resolution;
			glm::mat4 mvp;
		};
		struct MouseOver
		{
			Event::Primitive::MouseOver& event;
			Position                     position;
		};
	}
}

#endif
