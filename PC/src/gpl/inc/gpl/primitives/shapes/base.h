#ifndef _GPL_PRIMITIVE_SHAPE_BASE_H_
#define _GPL_PRIMITIVE_SHAPE_BASE_H_

#include <gra/math.h>
#include <gra/renderer.h>
#include <gra/graphics_objects/program.h>
#include <gra/graphics_objects/vertex_array.h>
#include <gra/graphics_objects/buffers/vertex.h>

#include "gpl/core.h"
#include "gpl/primitives/base.h"

namespace Gpl
{
	namespace Primitive
	{
		namespace Shape
		{
			class Base : public Primitive::Base
			{

				protected: // CONSTRUCTORS
					Base(const Gra::Math::vec4<float>& initial_color);

				public: // GETTERS
					const Gra::Math::vec4<float>& color(void) const;
				public: // SETTERS
					void color(const Gra::Math::vec4<float>& new_color);

				public: // FUNCTIONS
					virtual void draw(const Gra::Math::vec2<unsigned int>& resolution, const glm::mat4& parent_mvp) override;

				protected:
					Gra::Math::vec4<float> m_color;

				DECLARATION_MANDATORY_INTERFACE(Base)
			};

			// GETTERS
			DEFINITION_DEFAULT_GETTER(Base, color, const Gra::Math::vec4<float>&)
			// SETTERS
			DEFINITION_DEFAULT_SETTER(Base, color, const Gra::Math::vec4<float>&)

			// FUNCTIONS
			inline void Base::draw(const Gra::Math::vec2<unsigned int>& resolution, const glm::mat4& parent_mvp) { (void) resolution; (void) parent_mvp; }

			DEFINITION_MANDATORY(Base, other.m_color)
		}
	}
}

#endif
