#ifndef _GPL_CANVAS_H_
#define _GPL_CANVAS_H_

#include <vector>
#include <memory>

#include <gra/renderer.h>

#include "gpl/primitives/base.h"
#include "gpl/static_initializer.h"

namespace Gpl
{
	class Canvas : private StaticInitializer
	{
		private: // MEMBER VARIABLES
			std::vector<std::shared_ptr<Primitive::Base>> m_primitives;
		public: // STATIC VARIABLES
			static Gra::Renderer s_renderer;

		public:
			Canvas(void);
			~Canvas(void);

		public:
			void render(void);

		public: // OPERATORS
			Canvas& operator<<(Primitive::Base* primitive);
	};
}

#endif
