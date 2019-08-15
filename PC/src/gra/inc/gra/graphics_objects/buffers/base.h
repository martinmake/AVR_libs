#ifndef _GRA_GRAPHICS_OBJECT_BUFFER_BASE_H_
#define _GRA_GRAPHICS_OBJECT_BUFFER_BASE_H_

#include <inttypes.h>

#include "gra/glstd.h"
#include "gra/gldebug.h"

#include "gra/graphics_objects/base.h"

namespace Gra
{
	namespace GraphicsObject
	{
		namespace Buffer
		{
			class Base : public GraphicsObject::Base
			{
				protected:
					GLenum m_type;

				public:
					Base(void);
					Base(GLenum initial_type);

					Base(const Base&  other);
					Base(      Base&& other);

					virtual ~Base(void);

				public:
					void buffer_data(const void* data, size_t size);

				public:
					void   bind(void) const override;
					void unbind(void) const override;

				public:
					void copy(const Base&  other);
					void move(      Base&& other);
			};

			inline Base::Base(const Base&  other) : Base(other.m_type) { copy(          other ); }
			inline Base::Base(      Base&& other) : Base(other.m_type) { move(std::move(other)); }

			inline void Base::buffer_data(const void* data, size_t size) { bind(); glCall(glBufferData(m_type, size, data, GL_STATIC_DRAW)); }

			inline void Base::  bind(void) const { glCall(glBindBuffer(m_type, m_renderer_id)); }
			inline void Base::unbind(void) const { glCall(glBindBuffer(m_type, 0            )); }
		}
	}
}

#endif
