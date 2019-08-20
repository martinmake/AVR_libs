#ifndef _GRA_GRAPHICS_OBJECT_BUFFER_BASE_H_
#define _GRA_GRAPHICS_OBJECT_BUFFER_BASE_H_

#include <inttypes.h>

#include "gra/gra.h"
#include "gra/graphics_objects/base.h"

namespace Gra
{
	namespace GraphicsObject
	{
		namespace Buffer
		{
			class Base : public GraphicsObject::Base
			{
				protected: // CONSTRUCTORS
					Base(GLenum initial_type);

				public: // FUNCTIONS
					void buffer_data(const void* data, size_t size);

					void   bind(void) const override;
					void unbind(void) const override;

				protected:
					GLenum m_type;

				DECLARATION_MANDATORY_INTERFACE(Base)
			};

			// FUNCTIONS
			inline void Base::buffer_data(const void* data, size_t size) { bind(); glCall(glBufferData(m_type, size, data, GL_STATIC_DRAW)); }
			//
			inline void Base::  bind(void) const { glCall(glBindBuffer(m_type, m_renderer_id)); }
			inline void Base::unbind(void) const { glCall(glBindBuffer(m_type, 0            )); }

			DEFINITION_MANDATORY(Base, other.m_type)
		}
	}
}

#endif
