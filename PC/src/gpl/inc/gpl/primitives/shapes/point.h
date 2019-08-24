#ifndef _GPL_PRIMITIVE_SHAPE_POINT_H_
#define _GPL_PRIMITIVE_SHAPE_POINT_H_

#include "gpl/primitives/shapes/base.h"
#include "gpl/core.h"

namespace Gpl
{
	namespace Primitive
	{
		namespace Shape
		{
			class Point : public Primitive::Shape::Base
			{
				public: // CONSTRUCTORS
					Point(const Gra::Math::vec2<unsigned int>& initial_position, const Gra::Math::vec4<float>& initial_color, unsigned int initial_size);

				public: // FUNCTIONS
					void draw(const Gra::Math::vec2<unsigned int>& resolution, const glm::mat4& parent_mvp) override;

				public: // GETTERS
					const Gra::Math::vec2<unsigned int>& position(void) const;
							      unsigned int   size     (void) const;
				public: // SETTERS
					void position(const Gra::Math::vec2<unsigned int>& new_position);
					void size    (                      unsigned int   new_size    );

				private:
					Gra::Math::vec2<unsigned int> m_position;
					unsigned int m_size;

					DECLARATION_MANDATORY(Point)
			};

			// GETTERS
			DEFINITION_DEFAULT_GETTER(Point, position, const Gra::Math::vec2<unsigned int>&)
			DEFINITION_DEFAULT_GETTER(Point, size,                           unsigned int  )
			// SETTERS
			DEFINITION_DEFAULT_SETTER(Point, position, const Gra::Math::vec2<unsigned int>&)
			DEFINITION_DEFAULT_SETTER(Point, size,                           unsigned int  )

			DEFINITION_MANDATORY(Point, other.m_position, other.m_color, other.m_size)
		}
	}
}

#endif
