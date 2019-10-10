#ifndef _GPL_PRIMITIVE_SHAPE_POINT_H_
#define _GPL_PRIMITIVE_SHAPE_POINT_H_

#include "gpl/core.h"
#include "gpl/primitives/shapes/base.h"

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
					void draw(Data::Draw& data) override;

				public: // GETTERS
					const Position& position(void) const;
					unsigned int    size    (void) const;

					bool colides(const Position& position) const override;
				public: // SETTERS
					void position(const Position&    new_position);
					void size    (      unsigned int new_size    );

				private:
					Position     m_position;
					unsigned int m_size;

					DECLARATION_MANDATORY(Point)
			};

			// GETTERS
			DEFINITION_DEFAULT_GETTER(Point, position, const Position    &)
			DEFINITION_DEFAULT_GETTER(Point, size,           unsigned int )
			// SETTERS
			DEFINITION_DEFAULT_SETTER(Point, position, const Position    &)
			DEFINITION_DEFAULT_SETTER(Point, size,           unsigned int )

			DEFINITION_MANDATORY(Point, other.m_position, other.m_color, other.m_size)
		}
	}
}

#endif
