#ifndef _GRA_GRAPHICS_OBJECT_TEXTURE_PLANE_H_
#define _GRA_GRAPHICS_OBJECT_TEXTURE_PLANE_H_

#include <inttypes.h>

#include "gra/glstd.h"
#include "gra/gldebug.h"

#include "gra/graphics_objects/textures/base.h"

namespace Gra
{
	namespace GraphicsObject
	{
		namespace Texture
		{
			class Plane : public Texture::Base
			{
				private:
					int m_width,
					    m_height;

				public:
					Plane(                             unsigned int initial_slot = 0);
					Plane(const std::string& filepath, unsigned int initial_slot = 0);

					Plane(const Plane&  other);
					Plane(      Plane&& other);

					~Plane(void);

				public:
					void load(std::string filepath) override;

				public: // GETTERS
					int width (void) const;
					int height(void) const;

				private:
					void copy(const Plane&  other);
					void move(      Plane&& other);
				public:
					Plane& operator=(const Plane&  rhs);
					Plane& operator=(      Plane&& rhs);
			};

			// GETTERS
			inline          int Plane::width (void) const { return m_width;  }
			inline          int Plane::height(void) const { return m_height; }

			inline Plane::Plane(const Plane&  other) : Base() { copy(          other ); }
			inline Plane::Plane(      Plane&& other) : Base() { move(std::move(other)); }

			inline Plane& Plane::operator=(const Plane&  rhs) { copy(          rhs ); return *this; }
			inline Plane& Plane::operator=(      Plane&& rhs) { move(std::move(rhs)); return *this; }
		}
	}
}

#endif
