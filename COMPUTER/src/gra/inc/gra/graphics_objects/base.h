#ifndef _GRA_GRAPHICS_OBJECT_BASE_H_
#define _GRA_GRAPHICS_OBJECT_BASE_H_

#include "gra/core.h"

namespace Gra
{
	namespace GraphicsObject
	{
		class Base
		{
			protected: // CONSTRUCTORS
				Base(void);

			public: // FUNCTIONS
				virtual void   bind(void) const = 0;
				virtual void unbind(void) const = 0;

			public: // GETTERS
				unsigned int renderer_id(void) const;

			protected:
				unsigned int m_renderer_id;

			DECLARATION_MANDATORY_INTERFACE(Base)
		};

		inline unsigned int Base::renderer_id(void) const { return m_renderer_id; }
		DEFINITION_MANDATORY(Base, )
	}
}

#endif
