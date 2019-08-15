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
				protected:
					GLenum m_type;

					uint8_t* m_local_buffer;
					uint8_t  m_slot;

				public:
					Base(                     uint8_t initial_slot = 0);
					Base(GLenum initial_type, uint8_t initial_slot = 0);

					Base(const Base&  other);
					Base(      Base&& other);

					virtual ~Base(void);

				public:
					virtual void load(std::string filepath) { (void) filepath; }

				public:
					void   bind(void) const override;
					void unbind(void) const override;

				public: // GETTERS
					unsigned int slot(void) const;

				protected:
					void copy(const Base&  other);
					void move(      Base&& other);
			};

			// GETTERS
			inline unsigned int Base::slot(void) const { return m_slot; }

			inline Base::Base(const Base&  other) : Base() { copy(          other ); }
			inline Base::Base(      Base&& other) : Base() { move(std::move(other)); }

			inline void Base::  bind(void) const { glCall(glActiveTexture(GL_TEXTURE0 + m_slot)); glCall(glBindTexture(m_type, m_renderer_id)); }
			inline void Base::unbind(void) const { glCall(glActiveTexture(GL_TEXTURE0 + m_slot)); glCall(glBindTexture(m_type, 0            )); }
		}
	}
}

#endif
