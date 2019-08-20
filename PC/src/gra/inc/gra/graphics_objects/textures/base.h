#ifndef _GRA_GRAPHICS_OBJECT_TEXTURE_BASE_H_
#define _GRA_GRAPHICS_OBJECT_TEXTURE_BASE_H_

#include <inttypes.h>

#include "gra/glstd.h"
#include "gra/gldebug.h"

#include "gra/graphics_objects/base.h"

namespace Gra
{
	namespace GraphicsObject
	{
		namespace Texture
		{
			class Base : public GraphicsObject::Base
			{
				public: // CONSTRUCTORS
					Base(                     uint8_t initial_slot = 0);
					Base(GLenum initial_type, uint8_t initial_slot = 0);

				public: // GETTERS
					unsigned int slot(void) const;

				public: // FUNCTIONS
					virtual void load(std::string filepath) { (void) filepath; }

					void   bind(void) const override;
					void unbind(void) const override;

				protected:
					GLenum m_type;

					uint8_t* m_local_buffer;
					uint8_t  m_slot;

				DECLARATION_MANDATORY_INTERFACE(Base)
			};

			// GETTERS
			DEFINITION_DEFAULT_GETTER(Base, slot, unsigned int)

			// FUNCTIONS
			inline void Base::  bind(void) const { glCall(glActiveTexture(GL_TEXTURE0 + m_slot)); glCall(glBindTexture(m_type, m_renderer_id)); }
			inline void Base::unbind(void) const { glCall(glActiveTexture(GL_TEXTURE0 + m_slot)); glCall(glBindTexture(m_type, 0            )); }

			DEFINITION_MANDATORY(Base, other.m_type)
		}
	}
}

#endif
