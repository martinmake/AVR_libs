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
					Base(GLenum initial_type, size_t initial_size);
					Base(GLenum initial_type, const void* initial_data, size_t initial_size);

				public: // GETTERS
					size_t size(void) const;
				public: // SETTERS
					void size(size_t new_size);
					void data(const void* new_data, size_t new_size);
					void data(const void* new_data, size_t new_size, size_t offset);

				public: // FUNCTIONS
					void   bind(void) const override;
					void unbind(void) const override;

				protected:
					GLenum m_type;
					size_t m_size;

				DECLARATION_MANDATORY_INTERFACE(Base)
			};

			// FUNCTIONS
			inline void Base::  bind(void) const { glCall(glBindBuffer(m_type, m_renderer_id)); }
			inline void Base::unbind(void) const { glCall(glBindBuffer(m_type, 0            )); }

			// GETTERS
			inline size_t Base::size(void) const { return m_size; }
			// SETTERS
			inline void Base::size(size_t new_size) { m_size = new_size; data(nullptr, m_size); }
			inline void Base::data(const void* new_data, size_t new_size               ) { m_size = new_size; bind(); glCall(glBufferData   (m_type,         m_size, new_data, GL_STATIC_DRAW)); }
			inline void Base::data(const void* new_data, size_t     size, size_t offset) {                    bind(); glCall(glBufferSubData(m_type, offset,   size, new_data));                 }

			DEFINITION_MANDATORY(Base, other.m_type)
		}
	}
}

#endif
