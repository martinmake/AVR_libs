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
				public: // CONSTRUCTORS
					Plane(                             unsigned int initial_slot = 0);
					Plane(const std::string& filepath, unsigned int initial_slot = 0);

				public: // GETTERS
					int width (void) const;
					int height(void) const;

				public: // FUNCTIONS
					void load(std::string filepath) override;

				private:
					int m_width,
					    m_height;

				DECLARATION_MANDATORY(Plane)
			};

			// GETTERS
			DEFINITION_DEFAULT_GETTER(Plane, width,  int)
			DEFINITION_DEFAULT_GETTER(Plane, height, int)

			DEFINITION_MANDATORY(Plane, )
		}
	}
}

#endif
